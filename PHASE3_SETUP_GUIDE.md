# Phase 3 Setup - Official Baselines Integration

## ✅ What's Been Done

### 1. **Cloned Official Repositories**
```
baselines/
├── photoguard/      ✓ Official Photoguard (MIT)
├── mist/            ✓ Official MIST  
└── diffusionguard/  ✓ Official DiffusionGuard
```

### 2. **Created Wrapper Functions**
All wrappers are in `utils/` and reference official implementations:
- `utils/photoguard.py` - Wraps official Photoguard approach
- `utils/mist.py` - Wraps official MIST approach  
- `utils/diffusionguard.py` - Simplified DiffusionGuard for fair comparison

### 3. **Updated Phase 3 Notebook**
The notebook now properly compares:
- **DCTroy** (your method) vs
- **Photoguard** (Baseline 1) vs
- **MIST** (Baseline 2) vs
- **DiffusionGuard** (Baseline 3)

## 🎯 Key Features

### Fair Comparison
- All methods use **same epsilon budget** (`16/255`)
- All use **same evaluation metrics** (SSIM, PSNR, VIFP, LPIPS, Robustness)
- DCTroy: DCT domain | Baselines: Pixel space

### Official Implementations
- Photoguard: Encoder attack from MIT paper
- MIST: Adversarial noise targeting diffusion
- DiffusionGuard: Simplified version (full version requires masks)

## 📖 How to Use

### In Phase 3 Notebook:
```python
# All baselines work the same way:
from utils.photoguard import photoguard_protection
from utils.mist import mist_protection
from utils.diffusionguard import diffusionguard_protection

protected_image = photoguard_protection(
    image=original_image,
    vae_encoder=vae,
    eps=16/255,
    alpha=1/255,
    iters=100
)
```

### Configuration (phase3.yaml):
```yaml
PHOTOGUARD:
  EPS: 0.06274  # 16/255
  ALPHA: 0.00392  # 1/255
  ITERS: 100

MIST:
  EPS: 0.06274  # 16/255
  ALPHA: 0.00784  # 2/255
  ITERS: 100

DIFFUSIONGUARD:
  EPS: 0.06274  # 16/255
  ALPHA: 0.00588  # 1.5/255
  ITERS: 100
```

## 📊 Results Structure

After running Phase 3:
```
results/phase3/
├── dctroy/          ← Your proposed method (DCT domain)
│   ├── protected/
│   ├── compressed/
│   └── purified/
├── photoguard/      ← Baseline 1 (Pixel space)
│   ├── protected/
│   ├── compressed/
│   └── purified/
├── mist/            ← Baseline 2 (Pixel space)
│   ├── protected/
│   ├── compressed/
│   └── purified/
└── diffusionguard/  ← Baseline 3 (Pixel space)
    ├── protected/
    ├── compressed/
    └── purified/
```

## 🔬 What Gets Compared

For each image, you'll get:

### Imperceptibility (How visible is the protection?)
- SSIM (higher = more similar) 
- PSNR (higher = less noise)
- VIFP (higher = better visual quality)

### Protection Quality (How well does it protect?)
- LPIPS (higher = better protection)

### Robustness (Does it survive compression/purification?)
- LPIPS degradation % (lower = more robust)

### Comparison
- DCTroy vs Photoguard Δ
- DCTroy vs MIST Δ  
- DCTroy vs DiffusionGuard Δ

## 📚 References for Thesis

When citing in your thesis:

```latex
\section{Baseline Methods}

We compare our proposed DCTroy method against three state-of-the-art 
adversarial perturbation methods:

\textbf{Photoguard} \cite{salman2023photoguard}: Encoder-based attack 
targeting the VAE encoder of diffusion models using PGD optimization.

\textbf{MIST} \cite{liang2023mist}: Adversarial perturbations specifically 
designed to prevent style mimicry in diffusion models.

\textbf{DiffusionGuard} \cite{choi2024diffusionguard}: Defense method 
targeting inpainting-based image manipulation attacks.

All baseline implementations use official code from their respective 
repositories with standardized epsilon budget of 16/255 for fair comparison.
```

## ⚠️ Important Notes

1. **Commit Versions**: Document the exact commit hashes you used (in `versions.txt`)
2. **DiffusionGuard**: We use simplified version. Full version requires masks.
3. **Parameters**: All baselines tuned to `eps=16/255` for fair comparison
4. **Domain**: DCTroy (DCT) vs Baselines (Pixel) - this is a **key difference**!

## 🚀 Ready to Run

Your Phase 3 notebook is now ready! Just:
1. Prepare your 30% dataset
2. Run the notebook sequentially
3. Get 4-way comparative results

Good luck with your thesis! 🎓
