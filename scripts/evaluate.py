import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from models.swinir_arch import SwinIR  # SwinIR model architecture

# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert PIL images to normalized PyTorch tensors [0,1]
transform = transforms.ToTensor()

def evaluate_model(model_path, clean_dir, noisy_dir, output_dir="results/images", rgb=False):
    """
    Evaluates a pre-trained SwinIR model on a dataset of noisy and clean image pairs.

    Args:
        model_path (str): Path to the saved SwinIR model (.pth file).
        clean_dir (str): Directory containing clean ground truth images.
        noisy_dir (str): Directory containing noisy input images.
        output_dir (str): Directory to save the denoised images.
        rgb (bool): If True, processes images as RGB. Else, grayscale.

    Returns:
        avg (float): Average PSNR across all images.
        scores (list): List of tuples (filename, PSNR score).
    """
    in_ch = 3 if rgb else 1  # Set input channels: 3 for RGB, 1 for grayscale

    # Initialize the SwinIR model for denoising task
    model = SwinIR(
        upscale=1,                    # No upscaling for denoising
        in_chans=in_ch,              # Input channels
        img_size=64,                 # Dummy image size for architecture setup
        window_size=8,               # Window size for shifted window attention
        img_range=1.0,               # Pixel value range [0,1]
        depths=[6, 6, 6, 6],         # Number of transformer blocks in each stage
        embed_dim=96,                # Embedding dimension
        num_heads=[6, 6, 6, 6],      # Number of attention heads per stage
        mlp_ratio=4.0,               # Ratio for MLP hidden layer
        upsampler='',                # No upsampling
        resi_connection='1conv'     # Residual connection type
    ).to(device)

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(os.listdir(clean_dir))  # Get all filenames in clean image directory

    total_psnr = 0
    scores = []

    # Loop through each image in the dataset
    for fname in files:
        # Load clean and noisy image, convert to RGB or grayscale as specified
        clean_img = Image.open(os.path.join(clean_dir, fname)).convert("RGB" if rgb else "L")
        noisy_img = Image.open(os.path.join(noisy_dir, fname)).convert("RGB" if rgb else "L")

        # Apply transformation and move to device
        clean = transform(clean_img).unsqueeze(0).to(device)  # Add batch dimension
        noisy = transform(noisy_img).unsqueeze(0).to(device)

        # Predict denoised image using the model
        with torch.no_grad():
            out = model(noisy)

        # Convert model output and clean image to NumPy format for PSNR calculation
        out_np = out.squeeze().cpu().numpy()
        clean_np = clean.squeeze().cpu().numpy()

        # If RGB, convert from CHW to HWC format
        if out_np.ndim == 3:
            out_np = out_np.transpose(1, 2, 0)
            clean_np = clean_np.transpose(1, 2, 0)

        # Compute PSNR between clean and denoised image
        score = psnr(clean_np, out_np, data_range=1.0)
        total_psnr += score
        scores.append((fname, score))  # Store score for each image

        # Convert denoised output to image and save it
        if rgb:
            out_img = Image.fromarray((out_np * 255).clip(0, 255).astype("uint8"), mode="RGB")
        else:
            out_img = Image.fromarray((out_np * 255).clip(0, 255).astype("uint8"), mode="L")
        out_img.save(os.path.join(output_dir, fname))

        print(f"{fname} - PSNR: {score:.2f} dB")

    # Compute average PSNR across all images
    avg = total_psnr / len(files)
    print("\nAverage PSNR across all images: {:.2f} dB".format(avg))

    return avg, scores  # Return average and individual PSNR scores
