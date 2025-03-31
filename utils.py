import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch


class ImageDenoisingDataset(Dataset):
    def __init__(self, root_dir, patch_size=128):
        """
        Args:
            root_dir (str): Root directory, e.g., 'data/train'
            patch_size (int): Patch size to crop from images
        """
        self.clean_dir = os.path.join(root_dir, "clean")
        self.noisy_dir = os.path.join(root_dir, "noisy")
        self.filenames = sorted(os.listdir(self.clean_dir))
        self.patch_size = patch_size
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Load clean and noisy image pair by matching filename
        fname = self.filenames[idx]
        clean_img = Image.open(os.path.join(self.clean_dir, fname)).convert("RGB")
        noisy_img = Image.open(os.path.join(self.noisy_dir, fname)).convert("RGB")

        # Convert to tensors
        clean = self.transform(clean_img)
        noisy = self.transform(noisy_img)

        # Crop a random patch
        h, w = clean.shape[1:]
        if h < self.patch_size or w < self.patch_size:
            raise ValueError(
                f"Image {fname} is smaller than patch size {self.patch_size}"
            )

        y = torch.randint(0, h - self.patch_size + 1, (1,)).item()
        x = torch.randint(0, w - self.patch_size + 1, (1,)).item()
        clean_patch = clean[:, y : y + self.patch_size, x : x + self.patch_size]
        noisy_patch = noisy[:, y : y + self.patch_size, x : x + self.patch_size]

        return clean_patch, noisy_patch
