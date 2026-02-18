"""
MIST Implementation - Official Wrapper
Reference: https://github.com/psyker-team/mist
Paper: "Adversarial Example Does Good: Preventing Painting Imitation from Diffusion Models"

This implementation uses the original MIST approach with LinfPGDAttack and target_model.
Source: baselines/mist/mist_v3.py and baselines/mist/Masked_PGD.py
"""

import sys
from pathlib import Path
import numpy as np

# Add MIST baseline to path
BASELINES_DIR = Path(__file__).parent.parent / "baselines"
MIST_DIR = BASELINES_DIR / "mist"
sys.path.insert(0, str(MIST_DIR))

import torch
import torch.nn as nn
from typing import Any
from PIL import Image
import torchvision.transforms as T
from einops import rearrange

# Import from official MIST implementation
from Masked_PGD import LinfPGDAttack


class identity_loss(nn.Module):
    """
    Identity loss from MIST for advertorch compatibility.
    Source: baselines/mist/mist_v3.py
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x


class target_model_simplified(nn.Module):
    """
    Simplified target model from MIST for encoder-based attack.
    This uses semantic loss only (mode=0) for consistency with Photoguard.
    
    Source: Adapted from baselines/mist/mist_v3.py
    """
    def __init__(self, vae_encoder, condition: str = "a painting"):
        super().__init__()
        self.vae_encoder = vae_encoder
        self.condition = condition
        self.fn = nn.MSELoss(reduction="sum")

    def forward(self, x, components=False):
        """
        Compute semantic loss based on VAE encoding.
        For fair comparison with other baselines, we target the VAE encoder.
        """
        # Encode to latent space
        z = self.vae_encoder.encode(x).latent_dist.mean
        
        # Maximize latent norm (semantic attack)
        loss = z.norm()
        
        if components:
            return 0, loss
        return -loss  # Negative for maximization


def mist_protection(
    image: Image.Image,
    vae_encoder: Any,
    eps: float = 16/255,
    alpha: float = 2/255,
    iters: int = 100,
    device: str = "cuda",
) -> Image.Image:
    """
    Apply MIST protection using the original LinfPGDAttack implementation.
    This uses the official MIST PGD algorithm with targeted attack.
    
    Reference: https://github.com/psyker-team/mist
    Source: baselines/mist/mist_v3.py and baselines/mist/Masked_PGD.py
    
    Args:
        image: PIL Image to protect
        vae_encoder: VAE encoder model from Stable Diffusion
        eps: Maximum perturbation (L_inf bound) - typically 16/255
        alpha: Step size per iteration - typically 2/255
        iters: Number of PGD iterations (called 'steps' in MIST)
        device: Device to run on
        
    Returns:
        Protected PIL Image with MIST adversarial perturbation
    """
    # Prepare image using MIST's normalization: [0, 255] -> [-1, 1]
    img_array = np.array(image.resize((512, 512))).astype(np.float32) / 127.5 - 1.0
    img_array = img_array[:, :, :3]
    
    trans = T.ToTensor()
    data_source = torch.zeros([1, 3, 512, 512]).to(device)
    data_source[0] = trans(img_array).to(device)
    
    # Create target model (MIST's virtual model for semantic loss)
    net = target_model_simplified(vae_encoder, condition="a painting")
    net.eval()
    
    # Create identity loss function (MIST approach)
    fn = identity_loss()
    
    # Create label tensor (dummy, not used in loss computation)
    label = torch.zeros(data_source.shape).to(device)
    
    # Convert eps and alpha to MIST's format (for [-1, 1] range)
    # Original image range is [-1, 1], so we need: eps * 2
    eps_mist = eps * 2.0
    alpha_mist = alpha * 2.0
    
    # Apply MIST's LinfPGDAttack (targeted attack)
    attack = LinfPGDAttack(
        net, 
        fn, 
        eps=eps_mist, 
        nb_iter=iters,
        eps_iter=alpha_mist, 
        clip_min=-1.0, 
        clip_max=1.0,
        targeted=True  # MIST uses targeted attack
    )
    
    print(f"[MIST] Initial loss: {net(data_source, components=True)}")
    attack_output = attack.perturb(data_source, label)
    print(f"[MIST] Final loss: {net(attack_output, components=True)}")
    
    # Convert back to PIL image: [-1, 1] -> [0, 1] -> [0, 255]
    output = attack_output[0]
    save_adv = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0).detach()
    grid_adv = 255. * rearrange(save_adv, 'c h w -> h w c').cpu().numpy()
    
    protected_image = Image.fromarray(grid_adv.astype(np.uint8))
    
    return protected_image
