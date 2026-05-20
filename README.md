# Diffusion Model from Scratch in PyTorch

A complete implementation of a Denoising Diffusion Probabilistic Model (DDPM) and DDIM sampler built entirely from scratch using PyTorch.
The model is trained on the FashionMNIST dataset using an attention-enhanced U-Net architecture with sinusoidal timestep embeddings, EMA stabilization, and cosine variance scheduling.

## Generated Image
<img width="1184" height="1185" alt="epoch_100" src="https://github.com/user-attachments/assets/dc3b84f0-7e9a-44a9-879b-75ced75833c0" />

## Training Progress (GIF)
<img width="1184" height="1185" alt="training_progress" src="https://github.com/user-attachments/assets/3839b04e-8703-4343-a4ba-a8487287a169" />

## Project Structure
```text
diffusion-model/
│
├── generated_images/
│   ├── epoch_005.png
│   ├── epoch_010.png
│   └── ...
│
├── runs/
│   └── fashion_mnist_diffusion/
│
├── checkpoint.pth
├── ddim_fmnist.py
├── requirements.txt
├── README.md
└── training_progress.gif
```
## Requirements
```text
torch
torchvision
matplotlib
tensorboard
imageio
numpy
```
## References

- DDPM Paper: https://arxiv.org/abs/2006.11239  
  *Denoising Diffusion Probabilistic Models* (Ho et al., 2020)

- DDIM Paper: https://arxiv.org/abs/2010.02502  
  *Denoising Diffusion Implicit Models* (Song et al., 2021)
