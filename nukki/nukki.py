from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import time
import sys
import os
from safetensors.torch import load_file
import numpy as np

from models.birefnet import BiRefNet

from transformers import AutoModelForImageSegmentation

class Preprocess:
    @staticmethod
    def preprocess(path): 
        # Load the image
        image = Image.open(path)
        
        """
        rearrange 

        b c (hg h) (wg w) -> b (c hg wg) h w
        
        """

        block_h, block_w = 32, 32  # 블록 크기 (임의의 값, 조정 가능)
        original_width, original_height = image.size
        
        new_width = block_w * (original_width // block_w)
        new_height = block_h * (original_height // block_h)
        
        image = image.resize((new_width, new_height))


        transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Transform the resized image to tensor
        input_images = transform_image(image).unsqueeze(0).to('cuda')

        return image, input_images, (new_width, new_height)


class Model1: 
    def __init__(self):
        self.model = None
    
    def load_my_model(self, weight_path, device='cuda'):
        self.model = BiRefNet(bb_pretrained=False)
        state_dict = load_file(weight_path)
        self.model.load_state_dict(state_dict)
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        self.model.to(device)
        self.model.eval()

    def inf_mask_array(self, input_images):
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask_array = np.array(pred_pil, dtype=np.uint8)

        return mask_array


class Model2:
    def __init__(self): 
        self.model = None

    def load_my_model(self, pth_path, device='cuda'):
        self.model = AutoModelForImageSegmentation.from_pretrained(pth_path, trust_remote_code=True)
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        self.model.to(device)
        self.model.eval()

    def inf_mask_array(self, input_images):
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask_array = np.array(pred_pil, dtype=np.uint8)

        return mask_array


def combined_mask(mask1_array, mask2_array):    
    combined_mask_array = np.maximum(mask1_array, mask2_array)
    combined_mask = Image.fromarray(combined_mask_array, mode="L")
    return combined_mask 


def image_save(image, combined_mask):
    # Ensure the mask matches the image size
    combined_mask = combined_mask.resize(image.size)
    
    # Add alpha channel
    image.putalpha(combined_mask)
    
    # Save the image
    image.save("/home/sukhun/Downloads/BiRefNet/result/누끼/75.png")


def main():
    pre_image = Preprocess()
    
    # Preprocess input image
    image, input_images, resized_size = pre_image.preprocess("/home/sukhun/Downloads/BiRefNet/result/원본/75.jpg")

    # Load models
    model1 = Model1()
    model1.load_my_model('/home/sukhun/Downloads/BiRefNet/time/DIS-TR_TEs.safetensors')

    model2 = Model2()
    model2.load_my_model('briaai/RMBG-2.0')

    start_process_time = time.time()  # Start time

    # Generate masks
    mask1_array = model1.inf_mask_array(input_images)
    mask2_array = model2.inf_mask_array(input_images)

    # Combine masks
    mask_combined = combined_mask(mask1_array, mask2_array)

    end_process_time = time.time()  # End time


    # Save the image with combined mask
    image_save(image, mask_combined)

    print(f"Total processing time (excluding model load): {end_process_time - start_process_time:.2f} seconds")


if __name__ == "__main__":
    main()