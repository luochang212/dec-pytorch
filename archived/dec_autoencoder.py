import os

os.environ['LOKY_MAX_CPU_COUNT'] = "4"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment  # 替换废弃的sklearn函数

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
    "dims": [768, 512, 256, 64, 10],
    "n_clusters": 100,
    "pretrain_epochs": 100,
    "maxiter": 2000,
    "batch_size": 256,
    "update_interval": 100,
    "tol": 0.001,
    "alpha": 1.0,
    "save_dir": "../model"
}

# 1. 改进的自编码器结构
class Autoencoder(nn.Module):
    def __init__(self, dims):
        super().__init__()
        # 编码器
        encoder_layers = []
        for i in range(len(dims)-1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i != len(dims)-2:
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 解码器（对称结构）
        decoder_layers = []
        for i in reversed(range(len(dims)-1)):
            decoder_layers.append(nn.Linear(dims[i+1], dims[i]))
            if i != 0:
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        h = self.encoder(x)
        h = F.normalize(h, p=2, dim=1)  # 添加L2归一化
        return self.decoder(h)

# 2. 数值稳定的聚类层
class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, alpha=1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.Tensor(n_clusters, config["dims"][-1]))
        nn.init.xavier_normal_(self.clusters)
    
    def forward(self, x):
        # 稳定计算距离
        x = x.unsqueeze(1)  # [bs, 1, feat_dim]
        clusters = self.clusters.unsqueeze(0)  # [1, n_clusters, feat_dim]
        dist = torch.sum((x - clusters)**2, dim=2) / self.alpha  # [bs, n_clusters]
        
        # 数值稳定的soft分配
        q = 1.0 / (1.0 + dist)
        q = q ** ((self.alpha + 1.0) / 2.0)
        return q / torch.sum(q, dim=1, keepdim=True)

# 3. 改进的DEC模型
class DEC(nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoder = Autoencoder(config["dims"]).to(device)
        self.encoder = self.autoencoder.encoder
        self.clustering = ClusteringLayer(config["n_clusters"], config["alpha"])
        self.log_softmax = nn.LogSoftmax(dim=1)

    def target_distribution(self, q):
        """修正后的目标分布计算"""
        p = q**2 / torch.sum(q, dim=0)
        return (p.t() / torch.sum(p.t(), dim=1, keepdim=True)).t().detach()  # 关键修正
    
    def pretrain(self, data_loader):
        optimizer = optim.Adam(self.parameters())
        criterion = nn.MSELoss()
        self.train()
        
        for epoch in range(config["pretrain_epochs"]):
            total_loss = 0.0
            for idx, x in data_loader:
                x = x.to(device)
                optimizer.zero_grad()
                x_recon = self.autoencoder(x)
                loss = criterion(x_recon, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Pretrain Epoch {epoch+1}/{config['pretrain_epochs']}, Loss: {total_loss/len(data_loader):.4f}")
    
    def fit(self, X, y_true=None):
        # 数据准备（带索引）
        dataset = TensorDataset(torch.arange(len(X)), torch.from_numpy(X.astype(np.float32)))
        pretrain_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
        
        # 预训练阶段
        self.pretrain(pretrain_loader)
        
        # 初始化聚类中心
        with torch.no_grad():
            full_loader = DataLoader(dataset, batch_size=1024, shuffle=False)
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