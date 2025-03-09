import os
import torch

from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import DataLoader
from PIL import Image


MODEL_PATH = '../model/dinov2-base'
IMG_PATH = '../img'


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.processor(images=image, return_tensors="pt")['pixel_values'][0]


def get_img_path(dir_path):
    img_extensions = (".jpg", ".jpeg", ".png")
    img_list = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
            if f.lower().endswith(img_extensions)] 
    return [os.path.abspath(p) for p in img_list]


def load_model(device, model_path="facebook/dinov2-base"):
    # 加载模型和预处理器
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    return model, processor


def batch_inference(dataloader, model, device):
    embeddings = []
    device = torch.device(device)
    for pixel_values in dataloader:
        pixel_values = pixel_values.to(device)
        with torch.autocast(device_type=device.type), torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())

        del pixel_values, outputs
        torch.cuda.empty_cache()

    return torch.cat(embeddings, dim=0)


def main():
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    model, processor = load_model(device=device, model_path=MODEL_PATH)

    # 获取指定目录下所有图片文件的绝对路径
    image_paths = get_img_path(dir_path=IMG_PATH)

    # 加载数据
    dataset = ImageDataset(image_paths, processor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    embeddings = batch_inference(dataloader, model, device=device)
    print(embeddings)


if __name__ == '__main__':
    main()