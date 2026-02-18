import torch
import gc

from typing import Generator
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionImg2ImgPipeline,
)
from PIL import Image
from pathlib import Path

from .globals import DEVICE

STRENGTH = 0.5
GUIDANCE = 7.5
NUM_STEPS = 50
SEED = 9222
GENERATOR = torch.Generator(device=DEVICE).manual_seed(SEED)


def create_ip2p_pipeline(
    model_id: str = "timbrooks/instruct-pix2pix", local_path: str = None
) -> StableDiffusionInstructPix2PixPipeline:

    print("Using device:", DEVICE)

    if local_path is not None:
        return StableDiffusionInstructPix2PixPipeline.from_pretrained(
            local_path,
            torch_dtype=torch.float16,
            revision="fp16",
            local_path_only=True,
            safety_checker=None,
        ).to(DEVICE)

    return StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        revision="fp16",
        safety_checker=None,
    ).to(DEVICE)


@torch.no_grad()
def edit_image_with_ip2p(
    input_image: Image,
    instruction: str,
    ip2p_pipe: StableDiffusionInstructPix2PixPipeline,
    seed: int = SEED,
    num_inference_steps: int = NUM_STEPS,
    strength: float = STRENGTH,
    guidance_scale: float = GUIDANCE,
) -> Image:

    # Generate edited image
    edited_image = ip2p_pipe(
        instruction,
        image=input_image,
        seed=seed,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    ).images[0]

    torch.cuda.empty_cache()

    return edited_image


@torch.no_grad()
def edit_images_with_ip2p_batch(
    images: list[Image.Image],
    instructions: list[str],
    ip2p_pipe: StableDiffusionInstructPix2PixPipeline,
    num_inference_steps: int = NUM_STEPS,
    strength: float = STRENGTH,
    guidance_scale: float = GUIDANCE,
    seed: int = SEED,
) -> list[Image.Image]:

    assert len(images) == len(instructions)

    device = ip2p_pipe.device

    generators = torch.Generator(device=device).manual_seed(seed)

    with torch.autocast(device.type):
        result = ip2p_pipe(
            prompt=instructions,
            image=images,
            generator=generators,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

    outputs = result.images

    del result
    del generators
    torch.cuda.empty_cache()

    return outputs


@torch.no_grad()
def edit_images_with_ip2p_batch_looping(
    images: list[Image.Image],
    instructions: list[str],
    ip2p_pipe: StableDiffusionInstructPix2PixPipeline,
    seed: int = SEED,
    num_inference_steps: int = NUM_STEPS,
    strength: float = STRENGTH,
    guidance_scale: float = GUIDANCE,
) -> list[Image.Image]:

    assert len(images) == len(instructions), "images and instructions must align"

    outputs = []

    for img, instr in zip(images, instructions):
        result = ip2p_pipe(
            instr,
            image=img,
            seed=seed,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

        outputs.append(result.images[0])

        # Explicit cleanup (important for long runs)
        del result
        torch.cuda.empty_cache()

    return outputs


@torch.no_grad()
def edit_images_with_ip2p_batch_looping_generator(
    ip2p_pipe: StableDiffusionInstructPix2PixPipeline,
    instructions: list[str],
    images: list[Image.Image] | None = None,
    image_paths: list[str | Path] | None = None,
    seed: int = SEED,
    num_inference_steps: int = NUM_STEPS,
    strength: float = STRENGTH,
    guidance_scale: float = GUIDANCE,
) -> Generator[Image.Image, None, None]:

    if (images is None) == (image_paths is None):
        raise ValueError("Exactly one of `images` or `image_paths` must be provided.")

    if image_paths is not None:
        images = [Image.open(p) for p in image_paths]

    assert len(images) == len(instructions), "images and instructions must align"

    for i, (img, instr) in enumerate(zip(images, instructions)):
        result = ip2p_pipe(
            instr,
            image=img,
            seed=seed,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

        edited_img = result.images[0]

        yield edited_img

        # Explicit cleanup (important for long runs)
        del result, edited_img

        if i % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
