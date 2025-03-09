import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import utils

from datetime import datetime
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

if utils.in_jupyter():
    # 在 Jupyter 时 tqdm 的导入方式
    from tqdm.notebook import tqdm
else:
    # 在终端时 tqdm 的导入方式
    from tqdm import tqdm

os.environ['LOKY_MAX_CPU_COUNT'] = "4"

# 初始化 tensorboard writer
writer = SummaryWriter(f'runs/dec_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}')

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

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
    "dims": [768, 256, 32],
    "n_clusters": 100,
    "pretrain_epochs": 40,
    "maxiter": 300,
    "joint_train_iter": 150,
    "update_interval": 10,
    "batch_size": 256,
    "tol": 0.0001,
    "alpha": 1.0,
    "save_dir": "../model",
    "model_file": "dec_base.pth"
}

def add_noise(inputs, noise_type='mask', noise_factor=0.3):
    """
    噪声函数
    :param inputs: 输入张量
    :param noise_type: 'mask'(随机遮蔽) 或 'gaussian'(高斯噪声)
    :param noise_factor: 噪声强度
    """
    if noise_type == 'mask':
        # 随机遮蔽噪声
        mask = torch.rand_like(inputs) > noise_factor
        return inputs * mask
    elif noise_type == 'gaussian':
        # 高斯噪声（基于参数标准差动态调整）
        std = inputs.std(dim=0, keepdim=True)
        noise = torch.randn_like(inputs) * std * noise_factor
        return inputs + noise
    return inputs

class Autoencoder(nn.Module):
    """自编码器"""
    def __init__(self, dims):
        super().__init__()
        # 编码器
        encoder_layers = []
        for i in range(len(dims)-1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i != len(dims)-2:
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # 解码器
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

class ClusteringLayer(nn.Module):
    """聚类层"""
    def __init__(self, n_clusters, alpha=1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.Tensor(n_clusters, config["dims"][-1]))
        nn.init.xavier_normal_(self.clusters)

    def forward(self, x):
        # 计算距离
        x = x.unsqueeze(1)  # [bs, 1, feat_dim]
        clusters = self.clusters.unsqueeze(0)  # [1, n_clusters, feat_dim]
        dist = torch.sum((x - clusters)**2, dim=2) / self.alpha  # [bs, n_clusters]

        # 软分配
        power = -(self.alpha + 1.0) / 2.0
        q = (1.0 + dist).pow(power)

        eps = 0.0
        return q / (torch.sum(q, dim=1, keepdim=True) + eps)

class DEC(nn.Module):
    """DEC 模型"""
    def __init__(self):
        super().__init__()
        self.autoencoder = Autoencoder(config["dims"]).to(device)
        self.encoder = self.autoencoder.encoder
        self.clustering = ClusteringLayer(config["n_clusters"], config["alpha"])

    def target_distribution(self, q):
        """目标分布计算"""
        eps = 0.0
        numerator = q ** 2 / torch.sum(q, dim=0, keepdim=True)  # 按簇求和
        p = (numerator / (torch.sum(numerator, dim=1, keepdim=True) + eps)).detach()
        return p.detach()

    def pretrain(self, data_loader):
        optimizer = optim.Adam(self.autoencoder.parameters())
        criterion = nn.MSELoss()
        self.train()

        for epoch in range(config["pretrain_epochs"]):
            total_loss = 0.0
            for idx, x in data_loader:
                x = x.to(device)
                noisy_x = add_noise(x, noise_type='mask', noise_factor=0.3)
                optimizer.zero_grad()
                x_recon = self.autoencoder(noisy_x)  # 输入带噪声数据
                loss = criterion(x_recon, x)  # 重建目标仍为原始数据
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Pretrain Epoch {epoch+1}/{config['pretrain_epochs']}, Loss: {total_loss/len(data_loader):.4f}")

    def init_cluster_centers(self, full_loader, y_true=None):
        """初始化聚类中心"""
        features, indices = [], []
        with torch.no_grad():
            for idx, x in full_loader:
                features.append(self.encoder(x.to(device)).cpu())
                indices.append(idx)
            features = torch.cat(features).numpy()
            indices = torch.cat(indices).numpy()

            kmeans = KMeans(n_clusters=config["n_clusters"], n_init=20)
            y_pred = kmeans.fit_predict(features)

        # 打印初始化聚类中心后的 acc
        init_acc = acc(y_true, y_pred)
        print(f'init_acc: {init_acc}')

        return kmeans, y_pred, init_acc

    def train_epoch(self, cluster_loader, optimizer, p):
        """通过目标分布p引导聚类优化"""
        total_loss = 0.0
        self.train()
        for idx, x in cluster_loader:
            x = x.to(device)
            optimizer.zero_grad()

            # 前向计算
            q_batch = self.clustering(self.encoder(x))
            log_q = torch.log(q_batch)

            # 获取对应p值
            p_batch = p[idx].to(device).detach()

            # cosine_sim = F.cosine_similarity(q_batch, p_batch).mean()
            # print(f'cosine_sim: {cosine_sim}')

            # 计算损失
            loss = F.kl_div(log_q, p_batch, reduction='batchmean')
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, norm_type=2)

            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(cluster_loader)

    def fit(self, X, y_true=None):
        # 数据准备
        dataset = TensorDataset(torch.arange(len(X)), torch.from_numpy(X.astype(np.float32)))
        pretrain_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

        # 预训练阶段
        self.pretrain(pretrain_loader)

        # 初始化聚类中心
        full_loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        kmeans, y_pred, init_acc = self.init_cluster_centers(full_loader, y_true)
        self.clustering.clusters.data = torch.tensor(kmeans.cluster_centers_, device=device)

        # 准备聚类优化
        y_pred_last = y_pred.copy()

        # 最佳模型
        best_acc = init_acc
        bset_y_pred = y_pred
        best_model_state = self.state_dict()

        # 优化器
        optimizer = optim.Adam([
            {'params': self.encoder.parameters(), 'lr': 1e-5},
            {'params': self.clustering.parameters(), 'lr': 1e-4}
        ])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         patience=30,
                                                         factor=0.3,
                                                         min_lr=1e-7)

        # 分阶段控制参数
        freeze_encoder = True  # 初始冻结编码器
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 主训练循环
        cluster_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
        with tqdm(total=config["maxiter"], desc="Clustering") as pbar:
            for ite in range(config["maxiter"]):
                # 切换到联合优化阶段
                if ite == config["joint_train_iter"] and freeze_encoder:
                    print("\n=== Jointly optimize Autoencoder and ClusteringLayer ===")
                    for param in self.encoder.parameters():
                        param.requires_grad = True  # 解冻编码器参数
                    freeze_encoder = False

                    # 调整学习率
                    for param_group in optimizer.param_groups:
                        if 'encoder' in str(param_group['params'][0].shape):
                            param_group['lr'] = 1e-5  # 修改编码器学习率

                # 更新目标分布
                if ite % config["update_interval"] == 0:
                    with torch.no_grad():
                        q = self.clustering(self.encoder(torch.from_numpy(X).float().to(device)))
                        p = self.target_distribution(q)

                        # 计算聚类指标
                        y_pred = q.argmax(1).cpu().numpy()
                        if y_true is not None:
                            current_acc = acc(y_true, y_pred)
                            current_nmi = nmi(y_true, y_pred)
                            current_ari = ari(y_true, y_pred)
                            pbar.set_postfix(ACC=current_acc, NMI=current_nmi, ARI=current_ari)

                            if current_acc > best_acc:
                                best_acc = current_acc
                                bset_y_pred = y_pred
                                best_model_state = self.state_dict()
                                print(f'best_acc: {best_acc:.4f}')

                        # 检查收敛
                        delta_label = np.sum(y_pred != y_pred_last) / X.shape[0]
                        if ite > 0 and delta_label < config["tol"]:
                            print(f"\nConverged at iteration {ite}")
                            break
                        y_pred_last = y_pred.copy()

                # 批量训练
                avg_loss = self.train_epoch(cluster_loader, optimizer, p)
                if (ite + 1) % 50 == 0:
                    print(f"Clustering Iteration {ite + 1}/{config['maxiter']}, Loss: {avg_loss:.4f}")

                scheduler.step(avg_loss)

                pbar.update(1)

                # tensorboard writer
                writer.add_scalar('Loss/Clustering', avg_loss, ite)

        # 保存最佳模型
        torch.save(best_model_state, os.path.join(config['save_dir'], config['model_file']))

        return bset_y_pred

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