# 参考：https://github.com/luochang212/sentiment-analysis/blob/main/util.py

import io
import ast
import json
import base64
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from collections import Counter


BATCH_SIZE = 8
MAX_WORKERS = 2


def in_jupyter():
    try:
        from IPython import get_ipython
        return 'ZMQInteractiveShell' in str(get_ipython().__class__)
    except ImportError:
        return False
    except AttributeError:
        return False


def image_to_base64(image_path):
    """对图片进行base64编码"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def client(base64_images, api_url):
    """DINOv2 Embedding 生成服务 客户端"""
    response = requests.post(
        api_url,
        json={"base64_images": base64_images},
        timeout=30
    )
    response.raise_for_status()
    return response


# 参数 ebd_cols 定义哪些列存了 embedding
def embedding_df_to_csv(df, csv_path, ebd_cols: list):
    """将带有 embedding 的 DataFrame 存入 csv"""
    def ebd2str(embedding):
        if not isinstance(embedding, list):
            if isinstance(embedding, str):
                embedding = ast.literal_eval(embedding)
            else:
                embedding = embedding.tolist()
        return json.dumps(embedding)

    for col in ebd_cols:
        df[col] = df[col].apply(ebd2str)

    df.to_csv(csv_path, index=False)


def read_embedding_csv(csv_path, ebd_cols: list):
    """将带有 embedding 的 csv 读入 DataFrame"""
    df = pd.read_csv(csv_path)
    for col in ebd_cols:
        df[col] = df[col].apply(ast.literal_eval).apply(lambda e: np.array(e))

    return df


def to_base64(image):
    # 预处理图像
    image = preprocess_image(image)  

    # 编码为 Base64 字符串
    pil_img = Image.fromarray(image)
    buffer = io.BytesIO()  # 保存到内存缓冲区（格式可选 PNG/JPEG）
    pil_img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return img_base64


def split_list(lst, batch_size=BATCH_SIZE):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


def batch_processor(images, max_tries=2):
    base64_images = [to_base64(e) for e in images]
    for _ in range(max_tries):
        response = utils.client(base64_images, api_url=API_URL)
        if response.status_code == 200:
            embeddings = response.json().get('embeddings')
            if len(embeddings) != len(images):
                print('Error: len(embeddings) != len(images)')
                break
            return embeddings
    return [None * len(images)]


def gen_image_embed(images):
    image_batches = split_list(images)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = executor.map(batch_processor, image_batches)
        for result in futures:
            results.extend(result)

    assert len(results) == len(image_batches)
    return results


def plot_labels_distribution(labels):
    """绘制标签分布"""
    # 剔除 -1 标签（噪声点）
    valid_labels = [e for e in labels if e != -1]

    # 统计每个标签的出现次数
    label_counts = Counter(valid_labels)

    # 提取标签和对应的计数
    unique_labels = list(label_counts.keys())
    counts = list(label_counts.values())

    # 绘制柱状图
    plt.bar(unique_labels, counts)
    plt.xlabel('Cluster Labels')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of DBSCAN Cluster Labels')
    plt.xticks(rotation=45)
    plt.show()


def sum_top_100_categories(labels):
    """统计每个类别的出现次数"""
    # 剔除 -1 标签（噪声点）
    valid_labels = [e for e in labels if e != -1]

    category_counts = Counter(valid_labels)

    # 获取前100个最大类别（若不足100则取全部）
    top_100 = category_counts.most_common(100)
    
    # 计算样本数量总和
    total = sum(count for _, count in top_100)
    
    return total


def encode_classes(final_y_pred):
    """将类别列表重新编码为从0开始的连续整数，并返回新旧映射字典"""
    # 转换为 numpy 数组
    y_pred = np.array(final_y_pred)

    # 获取唯一类别及其出现次数
    unique_classes, counts = np.unique(y_pred, return_counts=True)
    
    # 按出现次数降序排列类别
    sorted_indices = np.argsort(-counts)
    sorted_classes = unique_classes[sorted_indices]
    
    # 创建映射字典（原类别 -> 新编码）
    mapping = {cls: idx for idx, cls in enumerate(sorted_classes)}
    
    # 使用向量化操作进行编码转换
    encoded = np.vectorize(mapping.get)(y_pred)

    return encoded.tolist(), mapping


def process_predictions(y_pred, y_true, cluster_number=100):
    """
    同步处理预测值和真实值
    1. 剔除预测值为 -1 的样本
    2. 保留预测值中前 100 高频类别的样本
    """
    assert len(y_pred) == len(y_true)

    # 转换为 numpy 数组
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    # 步骤1：同步过滤 -1
    valid_mask = y_pred != -1
    filtered_y_pred = y_pred[valid_mask]
    filtered_y_true = y_true[valid_mask]

    # 处理空数据情况
    if filtered_y_pred.size == 0:
        return [], []

    # 步骤2：统计类别频率
    unique_classes, counts = np.unique(filtered_y_pred, return_counts=True)
    sorted_classes = unique_classes[np.argsort(-counts)]  # 降序排列

    # 步骤3：选取前100高频类别（自动处理不足情况）
    top_classes = sorted_classes[:cluster_number]

    # 步骤4：保留对应样本
    keep_mask = np.isin(filtered_y_pred, top_classes)
    final_y_pred = filtered_y_pred[keep_mask]
    final_y_true = filtered_y_true[keep_mask]

    encoded_y_pred, mapping = encode_classes(final_y_pred)

    return final_y_pred, final_y_true, np.array(encoded_y_pred), mapping