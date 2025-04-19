# SwinIR Image Denoising using PyTorch

This project implements a SwinIR-based image denoising model using PyTorch. 

## Project Structure

```
├── models
│   └── swin_arch.py          # SwinIR model implementation
│
├── scripts
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Testing loop + evaluation
│   ├── dataset_utils.py      # Custom dataset loader
│
├── checkpoints/              # Model checkpoints saved here
├── datasets/                 # Dataset directory
└── results/                  # Output directory for results
```

## Dataset Structure

Organize your dataset like this:

```
datasets/
├── generate_noisy_data.py
├── BSD68/
│   ├── grayscale/
│   │   └── test001.png ...
│   └── rgb/
│       └── 3096.jpg ...
│
├── BSD400/
│   └── test_001.png ...
│
└── Test12/
    ├── images/
    │   └── 01.png ...
    └── noisy_gray_sigma15/
        └── 01.png ...
```

## Requirements

Install dependencies:
```bash
pip install torch torchvision einops scikit-image pillow matplotlib
```

## How to Run

### Step 1: Generate Noisy Dataset
```bash
cd datasets
python generate_noisy_data.py
```

### Step 2: Train

#### For Known Noise
```python
from scripts.train import train_known_noise

train_known_noise("datasets/BSD400", sigma=15, save_path="checkpoints/swinir_gray_sigma15.pth", rgb=False)
train_known_noise("datasets/BSD400", sigma=25, save_path="checkpoints/swinir_gray_sigma25.pth", rgb=False)
train_known_noise("datasets/BSD400", sigma=50, save_path="checkpoints/swinir_gray_sigma50.pth", rgb=False)
```

#### For Unknown/Blind Noise
```python
from scripts.train import train_blind_noise

train_blind_noise("datasets/BSD400", save_path="checkpoints/swinir_gray_blind.pth", rgb=False)
```

### Step 3: Test
```python
from scripts.evaluate import evaluate_model

evaluate_model(
    model_path="checkpoints/swinir_gray_sigma15.pth",
    clean_dir="datasets/Test12/images",
    noisy_dir="datasets/Test12/noisy_gray_sigma15",
    output_dir="results/images/gray_test12_sigma15",
    rgb=False
)
```

