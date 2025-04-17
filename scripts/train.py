import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# SwinIR model and custom datasets
from models.swinir_arch import SwinIR
from scripts.dataset_utils import KnownGaussianDataset, BlindGaussianDataset

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert PIL images to PyTorch tensors
transform = transforms.ToTensor()


# -----------------------------
# Function to build SwinIR model
# -----------------------------
def get_model(rgb):
    # Set input channels based on whether the image is RGB or grayscale
    in_ch = 3 if rgb else 1

    # Instantiate SwinIR model with parameters suitable for denoising
    model = SwinIR(
        img_size=64,  # Expected input image size
        patch_size=1,  # Patch size of 1 for denoising
        in_chans=in_ch,  # Number of input channels
        embed_dim=96,  # Dimension of embedding
        depths=[6, 6, 6, 6],  # Number of blocks in each Swin Transformer stage
        num_heads=[6, 6, 6, 6],  # Number of attention heads per stage
        window_size=8,  # Size of attention window
        mlp_ratio=4.0,  # MLP hidden dim ratio
        qkv_bias=True,  # Add bias to QKV
        drop_rate=0.0,  # Dropout rate
        attn_drop_rate=0.0,  # Attention dropout rate
        drop_path_rate=0.1,  # Drop path rate
        norm_layer=nn.LayerNorm,  # Normalization layer
        ape=False,  # Absolute positional encoding
        patch_norm=True,  # Apply normalization after patch embedding
        use_checkpoint=False,  # Disable gradient checkpointing
        img_range=1.0,  # Pixel value range
        upsampler="",  # No upsampling used in denoising
        resi_connection="1conv",  # Use 1 convolution for residual connection
    )
    return model.to(device)  # Move model to the selected device


# -----------------------------
# Training with Known Gaussian Noise
# -----------------------------
def train_known_noise(data_dir, sigma, save_path, rgb=False):
    # Load dataset with fixed sigma Gaussian noise
    dataset = KnownGaussianDataset(data_dir, sigma, transform, rgb=rgb)

    # Use smaller batch size for RGB due to memory
    batch_size = 2 if rgb else 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build and prepare model
    model = get_model(rgb)
    model.train()

    # Optimizer and loss function
    opt = torch.optim.Adam(model.parameters(), lr=0.0002)
    loss_fn = nn.MSELoss()

    # Train for 20 epochs
    for ep in range(20):
        total = 0
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)

            # Forward pass through the model
            out = model(noisy)

            # Ensure output matches target shape (due to padding in SwinIR)
            out = out[:, :, : clean.shape[2], : clean.shape[3]]

            # Compute MSE loss
            loss = loss_fn(out, clean)

            # Backpropagation
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        # Log average loss for the epoch
        print(
            f"Epoch {ep+1} | Sigma: {sigma} | Color: {'RGB' if rgb else 'Grayscale'} | Loss: {total / len(loader):.4f}"
        )

    # Save the trained model weights
    torch.save(model.state_dict(), save_path)


# -----------------------------
# Training with Blind Gaussian Noise
# -----------------------------
from tqdm import tqdm


def train_blind_noise(data_dir, save_path, rgb=False):
    # Load dataset with randomly sampled sigma per image
    dataset = BlindGaussianDataset(data_dir, transform=transform, rgb=rgb)

    # DataLoader with multi-threading
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # Build and prepare model
    model = get_model(rgb)
    model.train()

    # Optimizer and loss function
    opt = torch.optim.Adam(model.parameters(), lr=0.0002)
    loss_fn = nn.MSELoss()

    # Train for 5 epochs
    for ep in range(5):
        total = 0

        # Progress bar for real-time updates
        progress_bar = tqdm(
            loader, desc=f"Blind Epoch {ep+1} | Color: {'RGB' if rgb else 'Grayscale'}"
        )

        for noisy, clean in progress_bar:
            noisy, clean = noisy.to(device), clean.to(device)

            # Forward pass
            out = model(noisy)

            # Match output shape to ground truth
            out = out[:, :, : clean.shape[2], : clean.shape[3]]

            # Compute loss
            loss = loss_fn(out, clean)

            # Optimize model
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())

        # Log average epoch loss
        print(f"Finished Epoch {ep+1} | Avg Loss: {total / len(loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
