########### Already Generated ##################
from PIL import Image
import os
import torch
from torchvision import transforms

def add_noise_to_dataset(image_dir, output_dir, sigma, rgb=False):
    os.makedirs(output_dir, exist_ok=True)
    mode = "RGB" if rgb else "L"

    for fname in os.listdir(image_dir):
        img = Image.open(os.path.join(image_dir, fname)).convert(mode)
        img_tensor = transforms.ToTensor()(img)
        noise = torch.randn_like(img_tensor) * (sigma / 255.0)
        noisy = img_tensor + noise
        noisy_img = transforms.ToPILImage()(noisy.clamp(0., 1.))
        noisy_img.save(os.path.join(output_dir, fname))

# Grayscale
for sigma in [15, 25, 35, 50]:
    add_noise_to_dataset("datasets/BSD68/grayscale", f"datasets/BSD68/noisy_gray_sigma{sigma}", sigma, rgb=False)
    add_noise_to_dataset("datasets/Test12/images", f"datasets/Test12/noisy_gray_sigma{sigma}", sigma, rgb=False)

# RGB
for sigma in [35, 45]:
    add_noise_to_dataset("datasets/BSD68/rgb", f"datasets/BSD68/noisy_rgb_sigma{sigma}", sigma, rgb=True)