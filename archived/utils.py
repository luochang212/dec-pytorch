# 参考：https://github.com/luochang212/sentiment-analysis/blob/main/util.py

import io
import ast
import json
import base64
import requests
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from PIL import Image


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
