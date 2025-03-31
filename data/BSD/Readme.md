#  SwinIR Image Denoising using PyTorch


This project implements a SwinIR-based image denoising model using PyTorch. It supports training and testing on datasets with separate **clean** and **noisy** image pairs, such as BSD500 or real-world noisy datasets.

Project Structure

├── swinir_model.py       # SwinIR model implementation
├── train.py              # Training loop
├── test.py               # Testing loop + evaluation
├── utils.py              # Custom dataset loader
├── results/              # Output images from testing
└── data/                 # Input image dataset


##  Dataset Structure

Organize your dataset like this:
data/BSD/
├── train/ 
 ├── clean/  
     ├── img1.png 
 └── noisy/ 
    ├── img1.png 

├── test/ 
 ├── clean/  
     ├── img1.png 
 └── noisy/ 
    ├── img1.png  


##  Requirements

Install dependencies:

```bash
pip install torch torchvision einops scikit-image pillow matplotlib

OR

pip install -r requirements.txt


## How to run

### Train

python train.py


### Test

python test.py swinir_epoch20.pth


## Credit 
Based on the SwinIR paper (CVPRW 2021)  https://arxiv.org/abs/2108.10257
Inspired by official SwinIR repo  https://github.com/JingyunLiang/SwinIR






 
 


