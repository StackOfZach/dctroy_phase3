"""
Photoguard Implementation - Official Wrapper
Reference: https://github.com/MadryLab/photoguard
Paper: "Raising the Cost of Malicious AI-Powered Image Editing"

This implementation uses the original PGD-based encoder attack from the official Photoguard repository.
Source: baselines/photoguard/notebooks/demo_simple_attack_img2img.ipynb
"""

import sys
from pathlib import Path

# Add photoguard baseline to path
BASELINES_DIR = Path(__file__).parent.parent / "baselines"
PHOTOGUARD_DIR = BASELINES_DIR / "photoguard" / "notebooks"
sys.path.insert(0, str(PHOTOGUARD_DIR))

import torch
from tqdm import tqdm
from typing import Any
from PIL import Image
import torchvision.transforms as T

# Import utilities from official Photoguard implementation
from utils import preprocess as photoguard_preprocess


def pgd_photoguard(X, model, eps=0.1, step_size=0.015, iters=40, clamp_min=0, clamp_max=1, mask=None):
    """
    Original PGD implementation from Photoguard.
    Source: baselines/photoguard/notebooks/demo_simple_attack_img2img.ipynb
    
    Args:
        X: Input image tensor (normalized to [-1, 1])
        model: VAE encoder model
        eps: Maximum perturbation budget
        step_size: Initial step size for PGD
        iters: Number of PGD iterations
        clamp_min: Minimum clamp value
        clamp_max: Maximum clamp value
        mask: Optional mask for selective protection
        
    Returns:
        Adversarially perturbed image tensor
    """
    X_adv = X.clone().detach() + (torch.rand(*X.shape)*2*eps-eps).cuda()
    pbar = tqdm(range(iters))
    for i in pbar:
        # Linearly decaying step size
        actual_step_size = step_size - (step_size - step_size / 100) / iters * i  

        X_adv.requires_grad_(True)

        # Encoder attack: maximize latent norm
        loss = (model(X_adv).latent_dist.mean).norm()

        pbar.set_description(f"[Photoguard]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")

        grad, = torch.autograd.grad(loss, [X_adv])
        
        # PGD update
        X_adv = X_adv - grad.detach().sign() * actual_step_size
        X_adv = torch.minimum(torch.maximum(X_adv, X - eps), X + eps)
        X_adv.data = torch.clamp(X_adv, min=clamp_min, max=clamp_max)
        X_adv.grad = None    
        
        if mask is not None:
            X_adv.data *= mask
            
    return X_adv


def photoguard_protection(
    image: Image.Image,
    vae_encoder: Any,
    eps: float = 16/255,
    alpha: float = 1/255,
    iters: int = 100,
    device: str = "cuda",
) -> Image.Image:
    """
    Apply Photoguard protection using the original implementation.
    Uses PGD-based encoder attack that maximizes VAE latent norm.
    
    Reference: https://github.com/MadryLab/photoguard
    Source: baselines/photoguard/notebooks/demo_simple_attack_img2img.ipynb
    
    Args:
        image: PIL Image to protect
        vae_encoder: VAE encoder model from Stable Diffusion
        eps: Maximum perturbation (L_inf bound) - typically 16/255 = 0.06274
        alpha: Step size - typically 1/255 = 0.00392
        iters: Number of PGD iterations
        device: Device to run on
        
    Returns:
        Protected PIL Image with adversarial perturbation
    """
    to_pil = T.ToPILImage()
    
    # Use official Photoguard preprocessing (returns [-1, 1] normalized tensor)
    with torch.autocast('cuda'):
        X = photoguard_preprocess(image).half().cuda()
        
        # Apply original PGD attack from Photoguard
        adv_X = pgd_photoguard(
            X, 
            model=vae_encoder.encode, 
            clamp_min=-1, 
            clamp_max=1,
            eps=eps,
            step_size=alpha,
            iters=iters
        )
        
    # Convert back to PIL: [-1, 1] -> [0, 1] -> PIL
    adv_X = (adv_X / 2 + 0.5).clamp(0, 1)
    protected_image = to_pil(adv_X.squeeze(0).float().cpu())
    
    return protected_image
