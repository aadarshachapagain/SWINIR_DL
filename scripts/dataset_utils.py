import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class KnownGaussianDataset(Dataset):
    def __init__(self, clean_dir, sigma, transform, rgb=False, resize=(180, 180)):
        self.paths = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir)])
        self.sigma = sigma
        self.transform = transform
        self.rgb = rgb
        self.resize = resize

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB" if self.rgb else "L")
        if self.resize:
            img = img.resize(self.resize, Image.BICUBIC)
        clean = self.transform(img)
        noise = torch.randn_like(clean) * (self.sigma / 255.0)
        noisy = clean + noise
        
        return noisy.clamp(0., 1.), clean


class BlindGaussianDataset(Dataset):
    def __init__(self, clean_dir, patch_size=24, dataset_len=64000, noise_range=(0, 55), transform=None, rgb=False):
        self.paths = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir)]
        self.patch_size = patch_size
        self.len = dataset_len
        self.min_sigma, self.max_sigma = noise_range
        self.transform = transform
        self.rgb = rgb

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        path = np.random.choice(self.paths)
        img = Image.open(path).convert("RGB" if self.rgb else "L")
        img = self.transform(img)

        _, h, w = img.shape
        top = np.random.randint(0, h - self.patch_size)
        left = np.random.randint(0, w - self.patch_size)

        patch = img[:, top:top + self.patch_size, left:left + self.patch_size]
        sigma = np.random.uniform(self.min_sigma, self.max_sigma)
        noise = torch.randn_like(patch) * (sigma / 255.0)
        noisy = patch + noise
        return noisy.clamp(0., 1.), patch