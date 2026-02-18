import torch
import numpy as np
import matplotlib.pyplot as plt
from .globals import DEVICE


def get_frequency_map_8x8():
    """
    Create frequency map for 8x8 DCT block.
    Returns distance from DC component (0,0) - approximates frequency.

    Frequency increases diagonally from top-left to bottom-right:
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    [1.0, 1.4, 2.2, 3.2, 4.1, 5.1, 6.1, 7.1]
    ...
    [7.0, 7.1, 7.3, 7.6, 8.1, 8.6, 9.2, 9.9]
    """
    freq_map = torch.zeros(8, 8)
    for i in range(8):
        for j in range(8):
            # Euclidean distance from DC (0,0) approximates frequency
            freq_map[i, j] = np.sqrt(i**2 + j**2)
    return freq_map


# TODO: Need to fix the prevention of zero cause of diff jpeg now implemented
def create_frequency_weight_mask_v0(
    pattern="uniform", decay_strength=1.0, device=DEVICE
):
    """
    Create 8x8 weight mask for DCT coefficients based on frequency pattern.

    Args:
        pattern: One of ['uniform', 'low-pass', 'high-pass', 'mid-pass']
        decay_strength: Controls how aggressive the decay is (0.5-2.0 recommended)
        device: torch device

    Returns:
        8x8 weight mask (higher values = stronger perturbations allowed)
    """
    freq_map = get_frequency_map_8x8().to(device)
    max_freq = freq_map.max()

    # Normalize to [0, 1] range
    freq_norm = freq_map / max_freq

    if pattern == "uniform":
        weights = torch.ones(8, 8, device=device)

    elif pattern == "low-pass":
        # High weight at DC (top-left), low weight at high freq (bottom-right)
        weights = torch.exp(-decay_strength * freq_norm)

    elif pattern == "high-pass":
        # Low weight at DC, high weight at high frequencies
        weights = 1.0 - torch.exp(-decay_strength * freq_norm)
        weights = (weights * 0.9) + 0.1  # don't completely zero out low freqs

    elif pattern == "mid-pass":
        # Peak around mid-frequencies, decay towards both DC and highest freq
        center = 0.5  # Peak at middle frequency
        bandwidth = 0.3  # Width of the peak
        weights = torch.exp(-((freq_norm - center) ** 2) / (2 * bandwidth**2))
        weights = (weights * 0.9) + 0.1  # don't completely zero out low freqs
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return weights


def visualize_frequency_patterns():
    """Visualize all four frequency weighting patterns"""
    patterns = ["uniform", "low-pass", "high-pass", "mid-pass"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for idx, pattern in enumerate(patterns):
        mask = create_frequency_weight_mask(pattern, decay_strength=2.0, device="cpu")

        im = axes[idx].imshow(mask.numpy(), cmap="hot", vmin=0, vmax=1)
        axes[idx].set_title(
            f"{pattern.upper()} Weighting", fontsize=14, fontweight="bold"
        )
        axes[idx].set_xlabel("Horizontal Frequency →", fontsize=10)
        axes[idx].set_ylabel("Vertical Frequency →", fontsize=10)

        # Add colorbar
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

        # Add grid
        axes[idx].grid(True, alpha=0.3, color="white", linewidth=0.5)
        axes[idx].set_xticks(range(8))
        axes[idx].set_yticks(range(8))

    plt.tight_layout()
    # plt.savefig('frequency_weighting_patterns.png', dpi=150, bbox_inches='tight')

    print("Saved visualization to 'frequency_weighting_patterns.png'")
    plt.show()


def create_frequency_weight_mask(pattern="uniform", decay_strength=1.0, device="cuda"):
    """
    Create 8x8 weight mask normalized strictly to [0, 1].
    """
    freq_map = get_frequency_map_8x8().to(device)
    max_freq = freq_map.max()

    # Base Normalized Frequency [0, 1]
    freq_norm = freq_map / max_freq

    if pattern == "uniform":
        return torch.ones(8, 8, device=device)

    # 1. Generate Raw Curves
    elif pattern == "low-pass":
        # Exp decay
        weights = torch.exp(-decay_strength * freq_norm)

    elif pattern == "high-pass":
        # Inverted Exp decay
        weights = 1.0 - torch.exp(-decay_strength * freq_norm)

    elif pattern == "mid-pass":
        # Gaussian Bump
        center = 0.5
        bandwidth = 0.25  # Slightly tighter to ensure contrast
        weights = torch.exp(-((freq_norm - center) ** 2) / (2 * bandwidth**2))

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    w_min = weights.min()
    w_max = weights.max()

    # Avoid division by zero for uniform case (though uniform is handled above)
    if w_max - w_min > 1e-6:
        weights = (weights - w_min) / (w_max - w_min)
    else:
        # Fallback if weights are all identical
        weights = torch.ones_like(weights)

    return weights
