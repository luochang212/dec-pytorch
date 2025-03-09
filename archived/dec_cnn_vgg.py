import os

os.environ['LOKY_MAX_CPU_COUNT'] = "4"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision

from sklearn.cluster import KMeans
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 评估指标函数
def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[row_ind[i], col_ind[i]] for i in range(len(row_ind))]) * 1.0 / y_pred.size

def nmi(y_true, y_pred):
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(y_true, y_pred)

def ari(y_true, y_pred):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(y_true, y_pred)

# 超参数配置
config = {
    "input_dim": 768,
    "reshape_dim": (12, 8, 8),  # 修正为 (channels, height, width)
    "conv_dims": [32, 64, 128],  # 卷积通道数
    "n_clusters": 10,
    "pretrain_epochs": 100,
    "maxiter": 2000,
    "batch_size": 128,
    "update_interval": 100,
    "tol": 0.001,
    "alpha": 1.0,
    "save_dir": "../model",
    "use_perceptual_loss": True
}

# 1. 改进的卷积自编码器（适配向量输入）
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器保持不变
        self.encoder = nn.Sequential(
            nn.Conv2d(config["reshape_dim"][0], config["conv_dims"][0], 3, padding=1),
            nn.BatchNorm2d(config["conv_dims"][0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(config["conv_dims"][0], config["conv_dims"][1], 3, padding=1),
            nn.BatchNorm2d(config["conv_dims"][1]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(config["conv_dims"][1], config["conv_dims"][2], 3, padding=1),
            nn.BatchNorm2d(config["conv_dims"][2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 修正解码器结构
        self.decoder = nn.Sequential(
            # 输入: [bs, 128, 1, 1]
            nn.ConvTranspose2d(config["conv_dims"][2], config["conv_dims"][1], 
                             kernel_size=4, stride=1),  # 输出: [bs, 64, 4, 4]
            nn.BatchNorm2d(config["conv_dims"][1]),
            nn.ReLU(),
            
            nn.ConvTranspose2d(config["conv_dims"][1], config["conv_dims"][0], 
                             kernel_size=4, stride=2, padding=1),  # 输出: [bs, 32, 8, 8]
            nn.BatchNorm2d(config["conv_dims"][0]),
            nn.ReLU(),
            
            nn.ConvTranspose2d(config["conv_dims"][0], config["reshape_dim"][0], 
                             kernel_size=3, stride=1, padding=1),  # 输出: [bs, 12, 8, 8]
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(-1, *config["reshape_dim"])
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        h_flat = F.normalize(h_flat, p=2, dim=1)
        recon = self.decoder(h)  # 输出形状 [bs, 12, 8, 8]
        recon_flat = recon.view(-1, config["input_dim"])  # [bs, 768]
        return recon_flat, h_flat

# 2. 适配向量输入的感知损失
class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(768, 3*32*32)
        # self.vgg = torchvision.models.vgg16(pretrained=True).features[:15]
        self.vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features[:15]
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, recon, orig):
        # 投影到图像空间
        recon_img = self.proj(recon).view(-1,3,32,32)
        orig_img = self.proj(orig).view(-1,3,32,32)
        # 计算特征损失
        return F.mse_loss(self.vgg(recon_img), self.vgg(orig_img))

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, feat_dim, alpha=1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.Tensor(n_clusters, feat_dim))
        nn.init.xavier_normal_(self.clusters)
    
    def forward(self, x):
        dist = torch.cdist(x, self.clusters)**2 / self.alpha
        q = 1.0 / (1.0 + dist)
        q = q ** ((self.alpha +1.0)/2.0)
        return q / q.sum(dim=1, keepdim=True)

class DEC(nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoder = ConvAutoencoder().to(device)
        self.encoder = lambda x: self.autoencoder(x)[1]  # 获取展平特征
        
        # 调整聚类层输入维度
        self.clustering = ClusteringLayer(
            config["n_clusters"], 
            config["conv_dims"][-1],  # 使用最后一个卷积层通道数
            config["alpha"]
        )
        
        # 损失函数
        self.perceptual_loss = VGGLoss().to(device) if config["use_perceptual_loss"] else None
        self.mse_loss = nn.MSELoss()

    def target_distribution(self, q):
        """修正后的目标分布计算"""
        p = q**2 / q.sum(dim=0, keepdim=True)  # 保持维度用于广播
        p = p / p.sum(dim=1, keepdim=True)      # 按行归一化
        return p.detach()  # 确保梯度截断
    
    def pretrain(self, data_loader):
        optimizer = optim.Adam(self.parameters())
        
        for epoch in range(config["pretrain_epochs"]):
            total_loss = 0.0
            for idx, x in data_loader:
                x = x.to(device)
                optimizer.zero_grad()
                
                # 前向计算
                recon, _ = self.autoencoder(x)
                
                # 组合损失
                loss = self.mse_loss(recon, x)
                if self.perceptual_loss:
                    loss += 0.3 * self.perceptual_loss(recon, x)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Pretrain Epoch {epoch+1}/{config['pretrain_epochs']}, Loss: {total_loss/len(data_loader):.4f}")

    # 保持原有fit函数结构（需确保encoder输出正确维度）
    def fit(self, X, y_true=None):
        # 数据准备（保持原始形式）
        dataset = TensorDataset(torch.arange(len(X)), torch.from_numpy(X.astype(np.float32)))
        pretrain_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

        # 预训练阶段
        self.pretrain(pretrain_loader)
        
        # 初始化聚类中心
        with torch.no_grad():
            full_loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
            features, indices = [], []
            for idx, x in full_loader:
                features.append(self.encoder(x.to(device)).cpu())
                indices.append(idx)
            features = torch.cat(features).numpy()
            indices = torch.cat(indices).numpy()
            
            kmeans = KMeans(n_clusters=config["n_clusters"], n_init=20)
            y_pred = kmeans.fit_predict(features)
            self.clustering.clusters.data = torch.tensor(kmeans.cluster_centers_, device=device)

        # 准备聚类优化
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
        y_pred_last = y_pred.copy()
        best_acc = 0.0

        # 主训练循环
        cluster_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
        with tqdm(total=config["maxiter"], desc="Clustering") as pbar:
            for ite in range(config["maxiter"]):
                # 更新目标分布
                if ite % config["update_interval"] == 0:
                    with torch.no_grad():
                        # q = self.clustering(self.encoder(torch.from_numpy(X).float().to(device)))
                        # p = self.target_distribution(q)

                        q_list = []
                        for idx, x in DataLoader(dataset, batch_size=1024, shuffle=False):
                            x = x.to(device)
                            q_batch = self.clustering(self.encoder(x))
                            q_list.append(q_batch)
                        q = torch.cat(q_list, dim=0)  # 分批次计算全量q
                        p = self.target_distribution(q)  # 使用修正后的目标分布

                        # 计算聚类指标
                        y_pred = q.argmax(1).cpu().numpy()
                        if y_true is not None:
                            current_acc = acc(y_true, y_pred)
                            current_nmi = nmi(y_true, y_pred)
                            current_ari = ari(y_true, y_pred)
                            pbar.set_postfix(ACC=current_acc, NMI=current_nmi, ARI=current_ari)
                            
                            if current_acc > best_acc:
                                best_acc = current_acc
                                torch.save(self.state_dict(), f"{config['save_dir']}/best_model.pth")
                        
                        # 检查收敛
                        delta_label = np.sum(y_pred != y_pred_last) / X.shape[0]
                        if delta_label < config["tol"]:
                            print(f"\nConverged at iteration {ite}")
                            break
                        y_pred_last = y_pred.copy()
                
                # 批量训练
                for idx, x in cluster_loader:
                    x = x.to(device)
                    optimizer.zero_grad()
                    
                    # 前向计算
                    z = self.encoder(x)
                    q_batch = self.clustering(z)
                    log_q = self.log_softmax(q_batch)
                    
                    # 获取对应p值
                    p_batch = p[idx].to(device)
                    
                    # 计算损失
                    loss = F.kl_div(log_q, p_batch, reduction='batchmean')
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                pbar.update(1)

        return y_pred


# 使用示例
if __name__ == "__main__":
    # 生成示例数据（10000个768维向量）
    X = np.random.randn(10000, 768).astype(np.float32)
    y = np.random.randint(0, 10, (10000,))  # 假设真实标签
    
    # 初始化并训练模型
    dec = DEC().to(device)
    cluster_ids = dec.fit(X, y_true=y)
    
    # 输出最终结果
    print("\nFinal Clustering Results:")
    print(f"ACC: {acc(y, cluster_ids):.4f}")
    print(f"NMI: {nmi(y, cluster_ids):.4f}")
    print(f"ARI: {ari(y, cluster_ids):.4f}")
