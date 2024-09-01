import os
import numpy as np
from PIL import Image
import torch
import sys
import torchvision.transforms as transforms
sys.path.append('/data1/fyw/OASG-NET/fin/src/')
from unet import Unet

class BatchImageProcessor:
    def __init__(self, model, input_folder, output_folder, batch_size=4):
        self.model = model
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def process_images(self):
        image_files = [f for f in os.listdir(self.input_folder) if f.endswith('.png')]
        print(f"Total number of image files: {len(image_files)}")

        num_batches = len(image_files) // self.batch_size
        
        for batch_idx in range(num_batches):
            print(f"Processing batch {batch_idx + 1} of {num_batches}")
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_files = image_files[start_idx:end_idx]
            
            # Load and preprocess batch of images
            batch_tensors = []
            for file in batch_files:
                image_path = os.path.join(self.input_folder, file)
                image = Image.open(image_path).convert("RGB")
                input_tensor = self.transform(image).unsqueeze(0)
                batch_tensors.append(input_tensor)
            
            batch_tensors = torch.cat(batch_tensors, dim=0)
            
            # Process the batch
            masks, _, _ = self.model.detect_image2(batch_tensors)
            
            # Save the results
            for idx, mask in enumerate(masks):
                mask_np = mask.squeeze().cpu().numpy()
                mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
                output_path = os.path.join(self.output_folder, batch_files[idx])
                mask_img.save(output_path)

if __name__ == '__main__':
    mask_model = Unet()
    processor = BatchImageProcessor(mask_model, "/data1/fyw/OASG-NET/fin/datasets/whn/whn_test_images/", "/data1/fyw/OASG-NET/fin/datasets/whn/whn_test_masks/", batch_size=1)
    processor.process_images()