import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# ---------------------------
# Dataset for Known Gaussian Noise Level
# ---------------------------
class KnownGaussianDataset(Dataset):
    def __init__(self, clean_dir, sigma, transform, rgb=False, resize=(180, 180)):
        """
        Dataset class that adds a fixed (known) Gaussian noise level to each image.
        
        Args:
        - clean_dir (str): Directory with clean input images.
        - sigma (float): Standard deviation of Gaussian noise to be added.
        - transform: Torchvision transform to convert image to tensor and normalize.
        - rgb (bool): If True, use RGB images; otherwise use grayscale.
        - resize (tuple): Resize the image to this size.
        """
        self.paths = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir)])  # Sorted list of image paths
        self.sigma = sigma
        self.transform = transform
        self.rgb = rgb
        self.resize = resize

    def __len__(self):
        return len(self.paths)  # Total number of images

    def __getitem__(self, idx):
        # Load image and convert to RGB or grayscale
        img = Image.open(self.paths[idx]).convert("RGB" if self.rgb else "L")
        
        # Resize image if specified
        if self.resize:
            img = img.resize(self.resize, Image.BICUBIC)

        # Convert image to tensor
        clean = self.transform(img)

        # Generate Gaussian noise with specified sigma and add to image
        noise = torch.randn_like(clean) * (self.sigma / 255.0)
        noisy = clean + noise

        # Return noisy image and original clean image/clamp limits the value of a tensor to a specified limit
        return noisy.clamp(0., 1.), clean


# ---------------------------
# Dataset for Blind Gaussian Denoising
# ---------------------------
class BlindGaussianDataset(Dataset):
    def __init__(self, clean_dir, patch_size=24, dataset_len=64000, noise_range=(0, 55), transform=None, rgb=False):
        """
        Dataset class for blind denoising where noise level is randomly chosen for each image patch.
        
        Args:
        - clean_dir (str): Directory with clean images.
        - patch_size (int): Size of the square image patch to extract.
        - dataset_len (int): Number of patches to generate per epoch.
        - noise_range (tuple): Range (min, max) for random noise level (sigma).
        - transform: Torchvision transform to convert image to tensor.
        - rgb (bool): If True, use RGB images; otherwise use grayscale.
        """
        self.paths = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir)]  # List of all image paths
        self.patch_size = patch_size
        self.len = dataset_len
        self.min_sigma, self.max_sigma = noise_range
        self.transform = transform
        self.rgb = rgb

    def __len__(self):
        return self.len  # Total number of samples (patches)

    def __getitem__(self, idx):
        # Randomly select one image from the dataset
        path = np.random.choice(self.paths)
        img = Image.open(path).convert("RGB" if self.rgb else "L")

        # Apply transformations (e.g., to tensor)
        img = self.transform(img)

        # Get image dimensions (C, H, W)
        _, h, w = img.shape

        # Randomly select top-left corner for patch
        top = np.random.randint(0, h - self.patch_size)
        left = np.random.randint(0, w - self.patch_size)

        # Crop a random patch
        patch = img[:, top:top + self.patch_size, left:left + self.patch_size]

        # Sample a random noise level within specified range
        sigma = np.random.uniform(self.min_sigma, self.max_sigma)

        # Add Gaussian noise to patch
        noise = torch.randn_like(patch) * (sigma / 255.0)
        noisy = patch + noise

        # Return noisy patch and clean patch
        return noisy.clamp(0., 1.), patch
