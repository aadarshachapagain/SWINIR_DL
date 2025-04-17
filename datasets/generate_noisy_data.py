from PIL import Image
import os
import torch
from torchvision import transforms

def add_noise_to_dataset(image_dir, output_dir, sigma, rgb=False):
    """
    Adds Gaussian noise to all images in the given directory and saves the results.

    Parameters:
    - image_dir (str): Path to the directory with clean images.
    - output_dir (str): Path where noisy images will be saved.
    - sigma (float): Standard deviation of the Gaussian noise.
    - rgb (bool): If True, processes images in RGB. Otherwise, converts to grayscale.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    mode = "RGB" if rgb else "L"            # Set image mode (grayscale or color)

    for fname in os.listdir(image_dir):     # Iterate through all images in the directory
        img = Image.open(os.path.join(image_dir, fname)).convert(mode)  # Open and convert image
        img_tensor = transforms.ToTensor()(img)  # Convert PIL image to tensor [0, 1] range

        # Generate Gaussian noise and add to the image
        noise = torch.randn_like(img_tensor) * (sigma / 255.0)
        noisy = img_tensor + noise

        # Clamp pixel values to valid range and convert back to image
        noisy_img = transforms.ToPILImage()(noisy.clamp(0., 1.))
        noisy_img.save(os.path.join(output_dir, fname))  # Save noisy image

# Add grayscale noise at different standard deviations to BSD68 and Test12 datasets
for sigma in [15, 25, 35, 50]:
    add_noise_to_dataset("datasets/BSD68/grayscale", f"datasets/BSD68/noisy_gray_sigma{sigma}", sigma, rgb=False)
    add_noise_to_dataset("datasets/Test12/images", f"datasets/Test12/noisy_gray_sigma{sigma}", sigma, rgb=False)

# Add RGB noise at higher standard deviations (common in color image denoising tasks)
for sigma in [35, 45]:
    add_noise_to_dataset("datasets/BSD68/rgb", f"datasets/BSD68/noisy_rgb_sigma{sigma}", sigma, rgb=True)
