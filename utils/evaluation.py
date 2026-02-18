import numpy as np
import cv2
from PIL import Image

from cv2.ximgproc import guidedFilter

import torch
import lpips
from torchvision import transforms as T

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp

from typing import Tuple

_lpips_model = lpips.LPIPS(net="alex")
_lpips_model.eval()

_preprocess = T.Compose(
    [T.ToTensor(), T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]  # [0,1]  # -> [-1,1]
)

TARGET_LPIPS = 0.42
TARGET_LPIPS_DEGRADATION = 30.0  # percent
DCT_SHIELD_LPIPS = 0.684

TARGET_SSIM = 0.70
TARGET_PSNR = 29.5
TARGET_VIFP = 0.43

DCT_SHIELD_SSIM = 0.822
DCT_SHIELD_PSNR = 27.612
DCT_SHIELD_VIFP = 0.776


def _pil_to_np_float01(img: Image) -> np.ndarray:
    """
    Convert PIL Image to float32 numpy array in [0,1], shape [H,W,3]
    """
    img = img.convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def advclean(image: Image) -> Image:
    # Convert PIL Image to NumPy (OpenCV format)
    # display(image)
    image_np = np.array(image).astype(np.float32)

    y = image_np.copy()

    for _ in range(64):
        y = cv2.bilateralFilter(y, 5, 8, 8)

    for _ in range(4):
        y = guidedFilter(image_np, y, 4, 16)

    y_uint8 = y.clip(0, 255).astype(np.uint8)

    return Image.fromarray(y_uint8)


def jpeg_compress(image: Image, quality: int, verbose: bool = False) -> Image:
    """
    Compress an image using JPEG compression at the specified quality.

    Reports:
    - Raw image size (H × W × C bytes)
    - Compressed JPEG size (KB)
    """
    from io import BytesIO
    import numpy as np

    # Raw image size in bytes (RGB)
    image_np = np.asarray(image.convert("RGB"))
    raw_size_bytes = image_np.nbytes

    # JPEG compression
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    jpeg_bytes = buffer.getvalue()

    compressed_size_bytes = len(jpeg_bytes)

    if verbose:
        print(
            f"Raw image size: {raw_size_bytes / 1024:.2f} bytes, "
            f"Compressed JPEG size: {compressed_size_bytes / 1024:.2f} KB"
        )

    buffer.seek(0)
    compressed_image = Image.open(buffer)

    return compressed_image


def evaluate_lpips(img1: Image, img2: Image) -> float:
    """
    Evaluate LPIPS distance between two PIL images.
    """

    img1_tensor = _preprocess(img1).unsqueeze(0)
    img2_tensor = _preprocess(img2).unsqueeze(0)

    with torch.no_grad():
        distance = _lpips_model(img1_tensor, img2_tensor)

    return float(distance.item())


def evaluate_lpips_degradation(
    edited_img_src: Image,
    edited_img_protected: Image,
    edited_img_protected_tampered: Image,
    verbose: bool = False,
) -> Tuple[float, float]:

    original_lpips = evaluate_lpips(edited_img_src, edited_img_protected)
    tampered_lpips = evaluate_lpips(edited_img_src, edited_img_protected_tampered)

    if verbose:
        print(
            f"LPIPS Original: {original_lpips:.4f}, "
            f"LPIPS Tampered: {tampered_lpips:.4f}"
        )

    if original_lpips == 0:
        raise ValueError("original_lpips is zero; percentage drop undefined")

    degraded_lpips = original_lpips - tampered_lpips

    degradation_percent = (degraded_lpips / original_lpips) * 100.0

    return float(degradation_percent), float(tampered_lpips)


def evaluate_ssim(img1: Image, img2: Image) -> float:
    """
    Evaluate multichannel SSIM between two PIL images.
    """
    img1_np = _pil_to_np_float01(img1)
    img2_np = _pil_to_np_float01(img2)

    assert img1_np.shape == img2_np.shape, "Images must have same shape"

    return float(
        ssim(
            img1_np,
            img2_np,
            channel_axis=2,
            data_range=1.0,
        )
    )


def evaluate_psnr(img1: Image, img2: Image) -> float:
    """
    Evaluate PSNR between two PIL images.
    """
    img1_np = _pil_to_np_float01(img1)
    img2_np = _pil_to_np_float01(img2)

    assert img1_np.shape == img2_np.shape, "Images must have same shape"

    return float(
        psnr(
            img1_np,
            img2_np,
            data_range=1.0,
        )
    )


def evaluate_vifp(img1: Image, img2: Image) -> float:
    """
    Evaluate VIFp between two PIL images.
    """
    img1_np = _pil_to_np_float01(img1)
    img2_np = _pil_to_np_float01(img2)

    assert img1_np.shape == img2_np.shape, "Images must have same shape"

    return float(vifp(img1_np, img2_np))


def evaluate_imperceptibility(
    img_src: Image, img_protected: Image
) -> Tuple[float, float, float]:
    ssim_score = evaluate_ssim(img_src, img_protected)
    psnr_score = evaluate_psnr(img_src, img_protected)
    vifp_score = evaluate_vifp(img_src, img_protected)

    return ssim_score, psnr_score, vifp_score


def evaluate_robustness(
    edited_img_src: Image,
    edited_img_protected: Image,
    edited_img_protected_compressed: Image,
    edited_img_protected_purified: Image,
    verbose: bool = False,
) -> Tuple[float, float, float, float]:

    compressed_lpips_degradation, compressed_lpips = evaluate_lpips_degradation(
        edited_img_src,
        edited_img_protected,
        edited_img_protected_compressed,
        verbose=verbose,
    )
    purified_lpips_degradation, purified_lpips = evaluate_lpips_degradation(
        edited_img_src,
        edited_img_protected,
        edited_img_protected_purified,
        verbose=verbose,
    )

    if verbose:
        print(
            f"LPIPS Degradation Compression: {compressed_lpips_degradation:.2f}%\n"
            f"LPIPS Degradation Purification: {purified_lpips_degradation:.2f}%\n",
            f"DCT_SHIELD_LPIPS: {DCT_SHIELD_LPIPS}",
            f"TARGET_LPIPS_DEGRADATION: {TARGET_LPIPS_DEGRADATION}%",
            f"PASSED COMPRESSED: {TARGET_LPIPS_DEGRADATION >= compressed_lpips_degradation}",
            f"PASSED PURIFIED: {TARGET_LPIPS_DEGRADATION >= purified_lpips_degradation}",
            sep="\n",
        )

    return (
        float(compressed_lpips_degradation),
        float(compressed_lpips),
        float(purified_lpips_degradation),
        float(purified_lpips),
    )


def evaluate_relative_convergence():
    pass


def evaluate_peak_device_memory():
    pass


def evaluate_runtime():
    pass
