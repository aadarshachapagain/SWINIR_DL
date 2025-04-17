import os
from PIL import Image
import matplotlib.pyplot as plt

def show_comparison(noisy_path, denoised_path, clean_path, out_path):
    noisy = Image.open(noisy_path).convert("RGB")
    denoised = Image.open(denoised_path).convert("RGB")
    clean = Image.open(clean_path).convert("RGB")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(noisy)
    ax[0].set_title("Noisy")
    ax[0].axis("off")

    ax[1].imshow(denoised)
    ax[1].set_title("SwinIR")
    ax[1].axis("off")

    ax[2].imshow(clean)
    ax[2].set_title("Clean")
    ax[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def visualize_all(noisy_dir, denoised_dir, clean_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    files = sorted(os.listdir(clean_dir))

    for f in files:
        noisy_img = os.path.join(noisy_dir, f)
        denoised_img = os.path.join(denoised_dir, f)
        clean_img = os.path.join(clean_dir, f)
        out_img = os.path.join(save_dir, f"compare_{f}")

        show_comparison(noisy_img, denoised_img, clean_img, out_img)