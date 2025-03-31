from PIL import Image
import matplotlib.pyplot as plt

denoised = Image.open("results/output_2.png")
noisy = Image.open("results/noisy_2.png")
clean = Image.open("results/clean_2.png")

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(noisy)
plt.title("Noisy")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(denoised)
plt.title("Denoised")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(clean)
plt.title("Clean")
plt.axis("off")
plt.tight_layout()
plt.show()
