import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import utils

from typing import Optional
from datetime import datetime
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

if utils.in_jupyter():
    # 在 Jupyter 时 tqdm 的导入方式
    from tqdm.notebook import tqdm
else:
    # 在终端时 tqdm 的导入方式
    from tqdm import tqdm


# 参数配置
TRAIN_CSV_PATH = './data/train_embed_label.csv'
TEST_CSV_PATH = './data/test_embed_label.csv'

os.environ["LOKY_MAX_CPU_COUNT"] = "4"


# 模型超参数配置
config = {
    "dims": [768, 256, 32],
    "n_clusters": 100,
    "pretrain_epochs": 50,
    "soft_dist_epochs": 100,
    "update_interval": 10,
    "batch_size": 256,
    "tol": 0.001,
    "alpha": 1.0,
    "save_dir": "./model",
    "args_model_file": "dec_args.pth",
    "full_model_file": "dec_full.pth"
}


def load_embed_data(csv_path):
    """从 csv 文件加载 embedding 数据"""
    df = utils.read_embedding_csv(csv_path=csv_path,
                                  ebd_cols=['embeddings'])
    X = np.array(df['embeddings'].tolist())
    y_true = df['labels'].values
    return X, y_true


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


def add_noise(inputs, noise_type='mask', noise_factor=0.3):
    """
    噪声函数
    :param inputs: 输入张量
    :param noise_type: 'mask' or 'gaussian'
    :param noise_factor: 噪声强度
    """
    if noise_type == 'mask':
        # 掩蔽噪声
        mask = torch.rand_like(inputs) > noise_factor
        return inputs * mask
    elif noise_type == 'gaussian':
        # 高斯噪声
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


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    目标分布
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


class ClusterAssignment(nn.Module):
    """软分配"""
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.cluster_number,
                                                  self.embedding_dimension,
                                                  dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class DEC(nn.Module):
    """自编码器 + 软分配"""
    def __init__(
        self,
        cluster_number: int,
        hidden_dimension: int,
        encoder: torch.nn.Module,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None
    ):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param encoder: encoder to use
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(DEC, self).__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            cluster_number, self.hidden_dimension, alpha, cluster_centers
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        return self.assignment(self.encoder(batch))


def pretrain(autoencoder, data_loader, epochs, device, interval=10):
    """预训练自编码器"""
    optimizer = optim.Adam(autoencoder.parameters())
    criterion = nn.MSELoss()
    autoencoder.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for idx, x in data_loader:
            x = x.to(device)
            noisy_x = add_noise(x, noise_type='mask', noise_factor=0.3)
            optimizer.zero_grad()
            x_recon = autoencoder(noisy_x)  # 输入带噪声数据
            loss = criterion(x_recon, x)  # 重建目标仍为原始数据
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % interval == 0:
            print(f"Pretrain Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader):.4f}")
    return autoencoder


def init_cluster_centers(encoder, data_loader, n_clusters, device, y_true=None):
    """初始化聚类中心"""
    features, indices = [], []
    with torch.no_grad():
        for idx, x in data_loader:
            features.append(encoder(x.to(device)).cpu())
            indices.append(idx)
        features = torch.cat(features).numpy()
        indices = torch.cat(indices).numpy()
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(features)
    init_acc = None if y_true is None else acc(y_true, y_pred)
    return kmeans, y_pred, init_acc


def train_dec(model, data_loader, epochs, device, X, y_true=None, interval=10):
    """通过目标分布引导聚类优化"""

    # 记录最优模型
    best_model, best_acc = None, None

    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 1e-6},
        {'params': model.assignment.parameters(), 'lr': 1e-5}
    ])

    criterion = F.kl_div
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for idx, x in data_loader:
            x = x.to(device)
            optimizer.zero_grad()
            output = model(x)
            target = target_distribution(output).detach()
            loss = criterion(output.log(), target, reduction='batchmean')
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=1.0,
                                           norm_type=2)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        if (epoch + 1) % interval == 0:
            print(f"DEC Train Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        if y_true is not None:
            # 计算准确率
            with torch.no_grad():
                input = torch.from_numpy(X).float().to(device)
                y_pred = model(input).argmax(1).cpu().numpy()
            current_acc = acc(y_true, y_pred)

            # 更新最优模型
            if best_acc is None or current_acc > best_acc:
                best_model = copy.deepcopy(model)
                best_acc = current_acc
                print(f'===== best_acc: {best_acc:.4f} =====')

    return model if best_model is None else best_model, best_acc


def save_full_model(dec_model, config):
    """保存整个模型（结构+参数）"""
    full_model_path = os.path.join(config['save_dir'], config['full_model_file'])
    torch.save(dec_model, full_model_path)


def save_args_model(dec_model, config):
    """仅保存模型参数"""
    args_model_path = os.path.join(config['save_dir'], config['args_model_file'])
    torch.save(dec_model.state_dict(), args_model_path)


def load_full_model(model_path, device):
    """加载整个模型"""
    model = torch.load(model_path,
                       map_location=device,
                       weights_only=False)
    model.eval()
    return model


def load_args_model(model, model_path, device):
    """加载模型参数"""
    model.load_state_dict(
        torch.load(model_path,
                   map_location=device,
                   weights_only=True))
    model.eval()
    return model


def infer_embeddings(model,
                     embeddings: np.ndarray,
                     batch_size: int = 1024,
                     device: torch.device = torch.device('cpu')
    ) -> np.ndarray:
    """
    该函数用于对输入的嵌入向量进行聚类推理

    :param model: DEC 模型实例
    :param embeddings: numpy 数组，形状为 (N, 768)
    :param batch_size: 推理时的批量大小，默认为 1024
    :param device: 计算设备
    :return: 聚类标签，形状为 (N,)
    """
    model.eval()
    dataset = TensorDataset(torch.from_numpy(embeddings.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_labels = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            z = model.encoder(x)  # 编码器前向传播
            q = model.assignment(z)  # 聚类层计算分配概率
            labels = torch.argmax(q, dim=1).cpu().numpy()  # 获取聚类标签
            all_labels.append(labels)

    return np.concatenate(all_labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # 数据准备
    X, y_true = load_embed_data(TRAIN_CSV_PATH)

    dataset = TensorDataset(torch.arange(len(X)), torch.from_numpy(X.astype(np.float32)))
    pretrain_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # ======= 阶段一：训练降噪自编码器 =======
    # 实例化编码器
    auto_encoder = Autoencoder(config["dims"]).to(device)
    
    # 执行编码器预训练代码
    auto_encoder = pretrain(autoencoder=auto_encoder,
                            data_loader=pretrain_loader,
                            epochs=50,
                            device=device,
                            interval=config["update_interval"])

    # ======= 阶段二：初始化聚类中心 =======
    full_loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    kmeans, y_pred, init_acc = init_cluster_centers(encoder=auto_encoder.encoder,
                                                    data_loader=full_loader,
                                                    n_clusters=config["n_clusters"],
                                                    device=device,
                                                    y_true=y_true)

    print(f'init_acc: {init_acc}')

    # 代表聚类中心的特征向量
    cluster_centers = torch.tensor(kmeans.cluster_centers_,
                                   dtype=torch.float,
                                   requires_grad=True,
                                   device=device)

    # ======= 阶段三：训练 DEC =======
    # 实例化 DEC
    dec_model = DEC(
        cluster_number=config["n_clusters"],  # 预设的聚类数
        hidden_dimension=config["dims"][-1],  # 编码器输出维度
        encoder=auto_encoder.encoder,
        alpha=config["alpha"],
        cluster_centers=cluster_centers
    )

    data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    dec_model, dec_acc = train_dec(model=dec_model,
                                   data_loader=data_loader,
                                   epochs=config["soft_dist_epochs"],
                                   device=device,
                                   X=X,
                                   y_true=y_true,
                                   interval=config["update_interval"])

    # 保存最优模型
    save_full_model(dec_model, config)
    save_args_model(dec_model, config)

    # 计算指标
    y_pred = infer_embeddings(dec_model, X, device=device)

    print("\nFinal Clustering Results:")
    print(f"ACC: {acc(y_true, y_pred):.4f}")
    print(f"NMI: {nmi(y_true, y_pred):.4f}")
    print(f"ARI: {ari(y_true, y_pred):.4f}")


if __name__ == '__main__':
    main()
