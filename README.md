# dec-pytorch

深度嵌入聚类 (DEC) 算法实现，详见 [dec.py](./dec.py)

🚀 本项目的工作包括：

- [x] 用 DINOv2 模型生成图片 Embeddings
- [x] 用 FastAPI 开发 DINOv2 批量推理服务，支持分 batch 和 模型结果归一化
- [x] 训练 DEC 模型的三阶段：训练降噪自编码器、初始化聚类中心、训练 DEC
- [x] 开发集成的 DEC 训练框架，支持训算推存，详见 dec.py
- [x] 在我的数据集上，对比 DEC 与传统聚类算法的效果：与 K-Means 接近
- [x] 介绍 DEC 的创新点：软分配策略和目标分布优化
- [x] 在线学习探索：尝试两种思路，对 DEC 模型做小幅度的增量更新

## 一、使用 DINOv2 生成图片 Embedding

1. 从 huggingface 下载模型文件
2. 计算图片 Embedding
3. 批量计算图片 Embedding
    - 在 CPU 上批量推理
    - 在 GPU 上批量推理
4. 批量推理服务化
    - 启动服务端
    - 运行客户端

## 二、Embedding 数据准备

1. 下载 CIFAR-100 数据集
2. 图片转 Embedding

## 三、DEC 模型训练

1. 加载 Embedding 数据
2. 训练 DEC 模型
    - 初始化配置：初始化设备；定义评估指标函数
    - 定义降噪自编码器：支持加入掩蔽噪声或高斯噪声；添加了 L2 归一化
    - 定义主要组件：target_distribution, ClusterAssignment, DEC
    - 阶段一：训练降噪自编码器
    - 阶段二：初始化聚类中心
    - 阶段三：训练 DEC
    - 保存最优模型
    - 计算指标
3. 推理新数据
4. 评估

## 四、对比传统聚类算法

1. 加载数据
2. 评估函数
3. K-means 算法
4. DBSCAN 算法
5. 结论

## 五、深入学习 DEC 模型

1. 模型的创新点
    - 聚类中心初始化
    - 软分配策略
    - 目标分布优化
    - 模型训练
2. 训练阶段
3. 聚类标签匹配问题（匈牙利算法）
4. 模型推理和优化
    - 模型推理
    - 模型优化

## 六、探索：在线学习

1. 初次训练 DEC 模型
2. 生成新样本
3. 增量训练
    - 原模型在新数据集上的效果
    - 思路一：移动聚类中心
    - 思路二：重训练拟合目标分布的阶段
