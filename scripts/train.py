import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.swinir_arch import SwinIR
from scripts.dataset_utils import KnownGaussianDataset, BlindGaussianDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()


# Build the SwinIR model
def get_model(rgb):
    in_ch = 3 if rgb else 1
    model = SwinIR(
        img_size=64,
        patch_size=1,
        in_chans=in_ch,
        embed_dim=96,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        img_range=1.0,
        upsampler='',
        resi_connection='1conv'
    )
    return model.to(device)


def train_known_noise(data_dir, sigma, save_path, rgb=False):
    dataset = KnownGaussianDataset(data_dir, sigma, transform, rgb=rgb)
    batch_size = 2 if rgb else 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = get_model(rgb)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=0.0002)
    loss_fn = nn.MSELoss()

    for ep in range(20):
        total = 0
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)

            out = model(noisy)
            out = out[:, :, :clean.shape[2], :clean.shape[3]]  # crop output

            loss = loss_fn(out, clean)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {ep+1} | Sigma: {sigma} | Color: {'RGB' if rgb else 'Grayscale'} | Loss: {total / len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    

from tqdm import tqdm

def train_blind_noise(data_dir, save_path, rgb=False):
    dataset = BlindGaussianDataset(data_dir, transform=transform, rgb=rgb)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    model = get_model(rgb)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=0.0002)
    loss_fn = nn.MSELoss()

    for ep in range(5):
        total = 0
        progress_bar = tqdm(loader, desc=f"Blind Epoch {ep+1} | Color: {'RGB' if rgb else 'Grayscale'}")
        for noisy, clean in progress_bar:
            noisy, clean = noisy.to(device), clean.to(device)

            out = model(noisy)
            out = out[:, :, :clean.shape[2], :clean.shape[3]]

            loss = loss_fn(out, clean)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Finished Epoch {ep+1} | Avg Loss: {total / len(loader):.4f}")

    torch.save(model.state_dict(), save_path)