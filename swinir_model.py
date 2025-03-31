import torch
import torch.nn as nn

# A simple residual block with two convolutional layers and a skip connection
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),  # Preserve spatial size
            nn.ReLU(inplace=True),  # Activation function
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)   # Second conv layer
        )

    def forward(self, x):
        # Skip connection: output = input + F(input)
        return x + self.block(x)

# Main SwinIR model for denoising
class SwinIR(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dim=64, num_blocks=6):
        super().__init__()
        # Initial feature extraction from RGB input
        self.shallow_feat = nn.Conv2d(in_ch, dim, kernel_size=3, stride=1, padding=1)

        # Stack of residual blocks for deep feature learning
        self.blocks = nn.Sequential(*[ResidualBlock(dim) for _ in range(num_blocks)])

        # Final layer to map features back to RGB output
        self.upsample = nn.Conv2d(dim, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.shallow_feat(x)  # Shallow feature extraction
        x = self.blocks(x)        # Deep feature processing
        x = self.upsample(x)      # Output denoised image
        return x
