import torch
import torchvision.transforms as T

from diff_jpeg.encode import jpeg_encode as diff_jpeg_encode
from diff_jpeg.decode import jpeg_decode as diff_jpeg_decode
from diff_jpeg.rounding import *
from diff_jpeg.clipping import *
from diff_jpeg.utils import QUANTIZATION_TABLE_C, QUANTIZATION_TABLE_Y

from PIL import Image
from typing import Tuple


def jpeg_encode(
    image: torch.Tensor, jpeg_quality: int = 75, ste: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = image.device

    if ste:
        rounding_function = differentiable_rounding_ste
        floor_function = differentiable_floor_ste
        clipping_function = differentiable_clipping_ste
    else:
        rounding_function = differentiable_polynomial_rounding
        floor_function = differentiable_polynomial_floor
        clipping_function = differentiable_clipping

    jpeg_quality_tensor = torch.tensor([jpeg_quality], device=device)

    y_encoded, cb_encoded, cr_encoded = diff_jpeg_encode(
        image_rgb=image,
        jpeg_quality=jpeg_quality_tensor,
        quantization_table_c=QUANTIZATION_TABLE_C,
        quantization_table_y=QUANTIZATION_TABLE_Y,
        rounding_function=rounding_function,
        floor_function=floor_function,
        clipping_function=clipping_function,
    )

    return y_encoded, cb_encoded, cr_encoded


def jpeg_decode(
    y_encoded: torch.Tensor,
    cb_encoded: torch.Tensor,
    cr_encoded: torch.Tensor,
    H: int,
    W: int,
    jpeg_quality: int = 75,
    ste: bool = True,
) -> torch.Tensor:
    device = y_encoded.device

    if ste:
        floor_function = differentiable_floor_ste
        clipping_function = differentiable_clipping_ste
    else:
        floor_function = differentiable_polynomial_floor
        clipping_function = differentiable_clipping

    jpeg_quality_tensor = torch.tensor([jpeg_quality], device=device)

    decoded_image_tensor = diff_jpeg_decode(
        input_y=y_encoded,
        input_cb=cb_encoded,
        input_cr=cr_encoded,
        jpeg_quality=jpeg_quality_tensor,
        H=H,
        W=W,
        floor_function=floor_function,
        clipping_function=clipping_function,
        quantization_table_c=QUANTIZATION_TABLE_C,
        quantization_table_y=QUANTIZATION_TABLE_Y,
    )

    return decoded_image_tensor


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    # Convert PIL → Tensor in [0,1]
    image_tensor = T.ToTensor()(image)

    # Match torchvision.io.read_image behavior
    image_tensor = image_tensor * 255.0

    # Add batch dimension
    return image_tensor.unsqueeze(0)


def tensor_to_pil(image_tensor: torch.Tensor) -> Image:
    to_pil = T.ToPILImage()
    img_tensor = image_tensor[0]
    img_tensor = torch.clamp(img_tensor, 0, 255) / 255.0
    image = to_pil(img_tensor)
    return image
