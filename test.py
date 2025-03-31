import os
import torch
import torchvision.utils as vutils
from swinir_model import SwinIR
from utils import ImageDenoisingDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np


def tensor_to_numpy(img_tensor):
    """Convert torch tensor (1, 3, H, W) to numpy image (H, W, C) in [0,1]"""
    img = img_tensor.squeeze().detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW → HWC
    img = np.clip(img, 0, 1)
    return img


def test(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SwinIR().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load test dataset
    test_set = ImageDenoisingDataset(root_dir="data/BSD/test", patch_size=128)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    os.makedirs("results", exist_ok=True)

    total_psnr, total_ssim = 0.0, 0.0

    for i, (clean, noisy) in enumerate(test_loader):
        clean, noisy = clean.to(device), noisy.to(device)

        with torch.no_grad():
            output = model(noisy)

        # Save visual results (optional)
        vutils.save_image(output, f"results/output_{i}.png")
        vutils.save_image(noisy, f"results/noisy_{i}.png")
        vutils.save_image(clean, f"results/clean_{i}.png")

        # Convert to numpy for metric calculation
        clean_np = tensor_to_numpy(clean)
        output_np = tensor_to_numpy(output)

        # PSNR and SSIM
        psnr = peak_signal_noise_ratio(clean_np, output_np, data_range=1.0)
        ssim = structural_similarity(
            clean_np, output_np, channel_axis=2, data_range=1.0
        )

        total_psnr += psnr
        total_ssim += ssim

        print(f"Image {i}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")

    # Print average metrics
    num_images = len(test_loader)
    print(f"\nAverage PSNR: {total_psnr / num_images:.2f} dB")
    print(f"Average SSIM: {total_ssim / num_images:.4f}")


if __name__ == "__main__":
    test("swinir_epoch2.pth")  # Or your trained checkpoint path
