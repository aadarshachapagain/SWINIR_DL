import os
from PIL import Image
import matplotlib.pyplot as plt


# -----------------------------
# Visualize a single image comparison (noisy vs denoised vs clean)
# -----------------------------
def show_comparison(noisy_path, denoised_path, clean_path, out_path):
    # Load images and convert them to RGB format
    noisy = Image.open(noisy_path).convert("RGB")
    denoised = Image.open(denoised_path).convert("RGB")
    clean = Image.open(clean_path).convert("RGB")

    # Create a 1-row, 3-column plot for side-by-side comparison
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot noisy image
    ax[0].imshow(noisy)
    ax[0].set_title("Noisy")
    ax[0].axis("off")

    # Plot denoised output from SwinIR
    ax[1].imshow(denoised)
    ax[1].set_title("SwinIR")
    ax[1].axis("off")

    # Plot ground truth clean image
    ax[2].imshow(clean)
    ax[2].set_title("Clean")
    ax[2].axis("off")

    # Optimize spacing and save the plot
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# Visualize all images in a directory
# -----------------------------
def visualize_all(noisy_dir, denoised_dir, clean_dir, save_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Sort image filenames to ensure consistent order
    files = sorted(os.listdir(clean_dir))

    # Loop through all image files
    for f in files:
        # Construct full paths for noisy, denoised, and clean images
        noisy_img = os.path.join(noisy_dir, f)
        denoised_img = os.path.join(denoised_dir, f)
        clean_img = os.path.join(clean_dir, f)

        # Output path for comparison image
        out_img = os.path.join(save_dir, f"compare_{f}")

        # Generate side-by-side comparison plot and save it
        show_comparison(noisy_img, denoised_img, clean_img, out_img)
