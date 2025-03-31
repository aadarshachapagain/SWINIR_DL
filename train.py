import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from swinir_model import SwinIR
from utils import ImageDenoisingDataset
import os


def train():
    # Choose GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the SwinIR model
    model = SwinIR().to(device)

    # Define the optimizer (Adam) and loss function (MSE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()  # Mean Squared Error: common for image denoising

    # Load BSD500 training dataset with noise added on-the-fly
    train_set = ImageDenoisingDataset(root_dir="data/BSD/train", patch_size=128)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)

    # Loop over multiple epochs
    for epoch in range(2):
        model.train()  # Set model to training mode
        total_loss = 0

        # Iterate over all batches
        for clean, noisy in train_loader:
            clean, noisy = clean.to(device), noisy.to(device)

            # Forward pass: denoise the image
            output = model(noisy)

            # Compute loss between predicted clean and ground truth clean
            loss = criterion(output, clean)

            # Backward pass: update model weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average loss per epoch
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), f"swinir_epoch{epoch+1}.pth")


if __name__ == "__main__":
    train()
