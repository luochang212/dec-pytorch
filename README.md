# dec-pytorch

深度嵌入聚类 (DEC) 算法实现，详见 [dec.py](./dec.py)

## changelog

- [x] 用 DINOv2 模型生成图片 Embeddings
- [x] 用 FastAPI 开发 DINOv2 批量推理服务，支持分 batch 和 模型结果归一化
- [x] 训练 DEC 模型的三阶段：训练降噪自编码器、初始化聚类中心、训练 DEC
- [x] 开发集成的 DEC 训练框架，支持训算推存，详见 dec.py
- [x] 在我的数据集上，对比 DEC 与传统聚类算法的效果：与 K-Means 接近
- [x] 介绍 DEC 的创新点：软分配策略和目标分布优化
- [x] 在线学习探索：尝试两种思路，对 DEC 模型做小幅度的增量更新
