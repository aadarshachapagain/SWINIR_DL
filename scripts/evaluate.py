import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from models.swinir_arch import SwinIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()

def evaluate_model(model_path, clean_dir, noisy_dir, output_dir="results/images", rgb=False):
    in_ch = 3 if rgb else 1

    model = SwinIR(
        upscale=1,
        in_chans=in_ch,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=96,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=4.0,
        upsampler='',
        resi_connection='1conv'
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    files = sorted(os.listdir(clean_dir))

    total_psnr = 0
    scores = []

    for fname in files:
        clean_img = Image.open(os.path.join(clean_dir, fname)).convert("RGB" if rgb else "L")
        noisy_img = Image.open(os.path.join(noisy_dir, fname)).convert("RGB" if rgb else "L")

        clean = transform(clean_img).unsqueeze(0).to(device)
        noisy = transform(noisy_img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(noisy)

        out_np = out.squeeze().cpu().numpy()
        clean_np = clean.squeeze().cpu().numpy()

        if out_np.ndim == 3:
            out_np = out_np.transpose(1, 2, 0)
            clean_np = clean_np.transpose(1, 2, 0)

        score = psnr(clean_np, out_np, data_range=1.0)
        total_psnr += score
        scores.append((fname, score))

        if rgb:
            out_img = Image.fromarray((out_np * 255).clip(0, 255).astype("uint8"), mode="RGB")
        else:
            out_img = Image.fromarray((out_np * 255).clip(0, 255).astype("uint8"), mode="L")
            
        out_img.save(os.path.join(output_dir, fname))

        print(f"{fname} - PSNR: {score:.2f} dB")

    avg = total_psnr / len(files)
    print("\nAverage PSNR across all images: {:.2f} dB".format(avg))
    return avg, scores