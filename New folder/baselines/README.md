# Baseline Implementations for DCTroy Comparison

This directory contains official implementations of baseline methods used for comparative evaluation in Phase 3.

## 📚 Official Repositories

### 1. Photoguard

- **Repository**: https://github.com/MadryLab/photoguard
- **Paper**: "Raising the Cost of Malicious AI-Powered Image Editing" (ICML 2023)
- **Authors**: Salman et al., MIT
- **Method**: Encoder attack targeting VAE encoder with PGD
- **Citation**:
  ```bibtex
  @inproceedings{salman2023raising,
    title={Raising the Cost of Malicious AI-Powered Image Editing},
    author={Salman, Hadi and Khaddaj, Alaa and Leclerc, Guillaume and Ilyas, Andrew and Madry, Aleksander},
    booktitle={International Conference on Machine Learning},
    year={2023}
  }
  ```

### 2. MIST

- **Repository**: https://github.com/psyker-team/mist
- **Paper**: "Adversarial Example Does Good: Preventing Painting Imitation from Diffusion Models" (ICML 2023)
- **Authors**: Liang et al.
- **Method**: Low-level adversarial noise optimized for diffusion models
- **Citation**:
  ```bibtex
  @inproceedings{liang2023mist,
    title={Adversarial Example Does Good: Preventing Painting Imitation from Diffusion Models via Adversarial Examples},
    author={Liang, Chumeng and Wu, Xiaoyu and Hua, Yang and Zhang, Jiaru and Xue, Yiming and Song, Tao and Zheng, Ziyang and Ma, Ruhui and Guan, Haibing},
    booktitle={International Conference on Machine Learning},
    year={2023}
  }
  ```

### 3. DiffusionGuard

- **Repository**: https://github.com/choi403/DiffusionGuard
- **Paper**: "DiffusionGuard: A Robust Defense Against Diffusion-based Image Manipulation"
- **Authors**: Choi et al.
- **Method**: Adversarial perturbations targeting inpainting pipeline
- **Citation**:
  ```bibtex
  @article{choi2024diffusionguard,
    title={DiffusionGuard: A Robust Defense Against Diffusion-based Image Manipulation},
    author={Choi, Jeonghyeok and others},
    journal={arXiv preprint},
    year={2024}
  }
  ```

## 🔧 Setup Instructions

### Prerequisites

All baselines require:

- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended)

### Installation

1. **Clone all baseline repos** (already done if you're reading this):

   ```bash
   cd baselines/
   git clone https://github.com/MadryLab/photoguard
   git clone https://github.com/psyker-team/mist
   git clone https://github.com/choi403/DiffusionGuard diffusionguard
   ```

2. **Install Photoguard dependencies**:

   ```bash
   cd photoguard/
   pip install -r requirements.txt
   cd ..
   ```

3. **Install MIST dependencies**:

   ```bash
   cd mist/
   pip install -e .
   cd ..
   ```

4. **Install DiffusionGuard dependencies**:
   ```bash
   cd diffusionguard/
   pip install -r requirements.txt
   cd ..
   ```

## 📖 Usage in Phase 3

The wrapper functions in `utils/` automatically integrate these baseline implementations:

```python
from utils.photoguard import photoguard_protection
from utils.mist import mist_protection
from utils.diffusionguard import diffusionguard_protection

# All functions have the same interface:
protected_image = photoguard_protection(
    image=orig_image,
    vae_encoder=vae,
    eps=16/255,
    alpha=1/255,
    iters=100
)
```

## 🎯 Implementation Notes

### Photoguard

- Uses encoder attack (targeting VAE encoder)
- Also has diffusion attack mode (not used in our comparison)
- Parameters: `eps=16/255`, `alpha=1/255`, `iters=100`

### MIST

- Designed specifically for style mimicry attacks
- Uses zero initialization for perturbations
- Parameters: `eps=16/255`, `alpha=2/255`, `iters=100`

### DiffusionGuard

- **Full version**: Designed for inpainting attacks with masks
- **Our implementation**: Simplified encoder-based version for fair comparison
- For full DiffusionGuard capabilities, see `baselines/diffusionguard/main.py`
- Parameters: `eps=16/255`, `alpha=1.5/255`, `iters=100`

## ⚠️ Important Notes

1. **Fair Comparison**: All baselines are evaluated using the same `eps=16/255` budget for fair comparison with DCTroy.

2. **Evaluation Context**:
   - DCTroy operates in **DCT domain**
   - All baselines operate in **pixel space**

3. **DiffusionGuard Caveat**: The wrapper provides a simplified version. For full DiffusionGuard evaluation with inpainting masks, use the official `main.py`.

4. **Reproducibility**: Commit hashes of used versions are tracked in `versions.txt`.

## 📊 Expected Results Structure

After running Phase 3, results will be in:

```
results/phase3/
├── dctroy/          # DCTroy (proposed)
├── photoguard/      # Baseline 1
├── mist/            # Baseline 2
└── diffusionguard/  # Baseline 3
```

## 🐛 Troubleshooting

If you encounter import errors:

1. Ensure all baseline repos are cloned in `baselines/`
2. Check that dependencies are installed for each baseline
3. Verify Python path includes the baseline directories

For specific baseline issues, refer to their official repos.

## 📧 Contact

For questions about:

- **DCTroy implementation**: Contact thesis team
- **Baseline implementations**: Refer to original papers/repos
- **Integration issues**: Check Phase 3 notebook or contact thesis team

---

**Last Updated**: February 17, 2026
