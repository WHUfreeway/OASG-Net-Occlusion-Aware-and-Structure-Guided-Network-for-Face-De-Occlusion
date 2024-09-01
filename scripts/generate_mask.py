import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import sys
sys.path.append('/data1/fyw/OASG-NET/fin/src/')
from unet import Unet
import numpy as np

# 初始化模型
mask_model = Unet()

# 定义图像转换操作
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 定义输入和输出目录
input_dir = "/data1/fyw/datasets/WFLW_out/test_images/"
output_dir = "/data1/fyw/datasets/WFLW_out/test_masks/"


# 遍历输入目录中的所有.png文件
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        # 加载图像并转换为tensor
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path).convert("RGB")
        
            
        transform = transforms.Compose([
          transforms.Resize((256, 256)),
          transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0)  # 添加batch维度
        mask, mu1, mu2 = mask_model.detect_image2(input_tensor)

        # 4. 将预测的输出转换回图像格式
        mask = mask.squeeze().cpu().numpy()  # 移除batch维度并转换为numpy数组
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))

        output_path = os.path.join(output_dir, filename)
        mask_img.save(output_path)

        print(f"Processed {filename} and saved to {output_path}")

print("Processing complete!")

# 定义输入和输出目录
input_dir = "/data1/fyw/datasets/WFLW_out/train_images/"
output_dir = "/data1/fyw/datasets/WFLW_out/train_masks/"


# 遍历输入目录中的所有.png文件
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        # 加载图像并转换为tensor
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path).convert("RGB")
        
            
        transform = transforms.Compose([
          transforms.Resize((256, 256)),
          transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0)  # 添加batch维度
        mask, mu1, mu2 = mask_model.detect_image2(input_tensor)

        # 4. 将预测的输出转换回图像格式
        mask = mask.squeeze().cpu().numpy()  # 移除batch维度并转换为numpy数组
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))

        output_path = os.path.join(output_dir, filename)
        mask_img.save(output_path)

        print(f"Processed {filename} and saved to {output_path}")

print("Processing complete!")
