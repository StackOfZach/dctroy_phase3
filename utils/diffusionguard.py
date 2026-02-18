"""
DiffusionGuard Implementation - Official Wrapper (Simplified)
Reference: https://github.com/choi403/DiffusionGuard
Paper: "DiffusionGuard: A Robust Defense Against Diffusion-based Image Manipulation"

NOTE: Full DiffusionGuard is designed for inpainting attacks with mask generation.
The official implementation uses: generate_perturbation + attack_diffusionguard pipeline

For FAIR COMPARISON with pixel-space baselines (Photoguard, MIST), we use a
simplified encoder-based version that follows the same attack surface.

Official implementation files:
- baselines/diffusionguard/attacks/common.py (generate_perturbation)
- baselines/diffusionguard/attacks/attack_diffusionguard.py (attack_pipeline)
- baselines/diffusionguard/main.py (orchestration with masks)

Full version example: See notebooks/1diffusion_guard.ipynb for inpainting-based protection.
"""

import sys
from pathlib import Path

# Add DiffusionGuard baseline to path
BASELINES_DIR = Path(__file__).parent.parent / "baselines"
DIFFUSIONGUARD_DIR = BASELINES_DIR / "diffusionguard"
sys.path.insert(0, str(DIFFUSIONGUARD_DIR))

import torch
from tqdm import tqdm
from typing import Any
from PIL import Image
import torchvision.transforms as T


def diffusionguard_protection(
    image: Image.Image,
    vae_encoder: Any,
    eps: float = 16/255,
    alpha: float = 1.5/255,
    iters: int = 100,
    device: str = "cuda",
) -> Image.Image:
    """
    Apply DiffusionGuard protection using simplified encoder-based approach.
    
    IMPORTANT: Official DiffusionGuard targets inpainting with masks and uses:
      - generate_perturbation() from attacks/common.py
      - attack_diffusionguard() pipeline from attacks/attack_diffusionguard.py
      - Mask generation for selective perturbation
      - Gradient averaging over multiple batches
    
    For fair comparison with Photoguard and MIST (which are encoder-based),
    we use a simplified version targeting the same VAE encoder.
    
    To use full DiffusionGuard with inpainting:
      ```python
      from attacks import protect_image, diffusionguard_attack_pipeline
      adv = protect_image('diffusionguard', pipe, src_image, 
                         mask_image_list, mask_combined, mask_radius_list, config)
      ```
    
    Reference: https://github.com/choi403/DiffusionGuard
    Source: baselines/diffusionguard/attacks/
    
    Args:
        image: PIL Image to protect
        vae_encoder: VAE encoder model from Stable Diffusion
        eps: Maximum perturbation (L_inf bound) - typically 16/255
        alpha: Step size - typically 1.5/255
        iters: Number of optimization iterations
        device: Device to run on
        
    Returns:
        Protected PIL Image with adversarial perturbation
    """
    # Prepare image tensor
    preprocess = T.Compose([
        T.Resize(512),
        T.CenterCrop(512),
        T.ToTensor(),
    ])
    
    X = preprocess(image).unsqueeze(0).to(device)
    
    # Initialize with small random noise (similar to DiffusionGuard's approach)
    noise = torch.randn_like(X) * (eps / 2)
    X_adv = (X + noise).clamp(0, 1).detach()
    
    pbar = tqdm(range(iters), desc="[DiffusionGuard]")
    
    for i in pbar:
        # Adaptive step size (decays over iterations - similar to DiffusionGuard)
        actual_alpha = alpha * (1.0 - 0.5 * i / iters)
        
        X_adv.requires_grad_(True)
        
        # Normalize for VAE: [0, 1] -> [-1, 1]
        X_normalized = (X_adv * 2.0) - 1.0
        
        # Encoder attack: maximize latent norm
        # (Simplified from DiffusionGuard's attack_pipeline which uses UNet)
        latent_dist = vae_encoder.encode(X_normalized).latent_dist
        loss = latent_dist.mean.norm()
        
        pbar.set_description(f"[DiffusionGuard] Loss: {loss.item():.5f}")
        
        # Compute gradient
        grad = torch.autograd.grad(loss, X_adv)[0]
        
        # PGD update with sign gradient (DiffusionGuard uses sign in common.py)
        with torch.no_grad():
            # Sign-based gradient descent
            X_adv = X_adv - actual_alpha * grad.sign()
            
            # Project to epsilon ball
            delta = X_adv - X
            delta = torch.clamp(delta, -eps, eps)
            X_adv = X + delta
            
            # Clamp to valid pixel range
            X_adv = X_adv.clamp(0, 1)
        
        X_adv = X_adv.detach()
    
    # Convert back to PIL
    to_pil = T.ToPILImage()
    protected_image = to_pil(X_adv.squeeze(0).cpu())
    
    return protected_image
