# USAGE: python dinov2_client.py

import base64
import requests
import torch

API_URL = "http://localhost:8210/embeddings/"


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def client(base64_images):
    response = requests.post(
        API_URL,
        json={"base64_images": base64_images},
        timeout=30
    )
    response.raise_for_status()  # 触发 HTTP 错误状态异常
    return response.json()


if __name__ == "__main__":
    image_paths = [
        '../img/cat.jpg',
        '../img/book.jpg'
    ]
    base64_images = [image_to_base64(p) for p in image_paths]
    result = client(base64_images)
    embeddings = torch.Tensor(result['embeddings'])
    norms = torch.norm(embeddings, p=2, dim=1)

    print(f'embeddings.shape: {embeddings.shape}')
    print(f'norms: {norms}')
