import gc
import math
import time
from PIL import Image
from cv2 import transform
import torch
from tqdm import tqdm
from typing import Any, Tuple
from utils.data_controller import load_image_tensor, resize_image_tensor
from utils.jpeg_dct import jpeg_decode, jpeg_encode, tensor_to_pil
from pathlib import Path
import torch.nn.functional as F


def pgd_spatial(
    X,
    model,
    eps=0.1,
    step_size=0.015,
    iters=40,
    clamp_min=0.0,
    clamp_max=1.0,
    mask=None,
):
    """
    Projected Gradient Descent (PGD) attack in input space.

    - Random start within L_inf epsilon ball
    - Linearly decayed step size
    - Sign gradient update
    - Projection onto epsilon ball
    - Optional spatial mask
    """
    # --------------------------------------------------
    # 1. Random initialization within epsilon-ball
    # --------------------------------------------------
    noise = torch.empty_like(X).uniform_(-eps, eps)
    X_adv = (X + noise).detach()

    # --------------------------------------------------
    # 2. PGD optimization loop
    # --------------------------------------------------
    pbar = tqdm(range(iters))

    for i in pbar:
        # ---- Step size schedule (linear decay) ----
        actual_step_size = step_size - (step_size - step_size / 100) * (i / iters)

        # ---- Enable gradient tracking ----
        X_adv.requires_grad_(True)

        # ---- Forward + loss ----
        loss = model(X_adv).latent_dist.mean.norm()

        pbar.set_description(
            f"[Running attack] Loss: {loss.item():.5f} | step size: {actual_step_size:.4f}"
        )

        # ---- Compute gradient ----
        (grad,) = torch.autograd.grad(loss, X_adv, only_inputs=True)

        # --------------------------------------------------
        # 3. Gradient step (L_inf)
        # --------------------------------------------------
        with torch.no_grad():
            X_adv = X_adv - actual_step_size * grad.sign()

            # ---- Projection onto epsilon-ball ----
            X_adv = torch.max(torch.min(X_adv, X + eps), X - eps)

            # ---- Clamp to valid input range ----
            X_adv = torch.clamp(X_adv, clamp_min, clamp_max)

            # ---- Optional mask ----
            if mask is not None:
                X_adv = X_adv * mask

        X_adv = X_adv.detach()

    return X_adv


def pgd_dct(
    Y_q: torch.Tensor,
    Cb_q: torch.Tensor,
    Cr_q: torch.Tensor,
    vae_encoder: Any,
    H: int = 512,
    W: int = 512,
    jpeg_q: int = 95,
    eps: float = 1.0,
    alpha: float = 0.1,
    iters: int = 1000,
    image_id: str = "unknown",
    frequency_weight: torch.Tensor = None,
    checkpoints: list[int] = None,
    checkpoint_save_path: str | Path = "results/checkpoints",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    device = Y_q.device
    base_checkpoint_path = Path(checkpoint_save_path)

    # 1. Initialize perturbations for ALL channels
    delta_Y = torch.zeros_like(Y_q, device=device, requires_grad=True)
    delta_Cb = torch.zeros_like(Cb_q, device=device, requires_grad=True)
    delta_Cr = torch.zeros_like(Cr_q, device=device, requires_grad=True)

    saved_checkpoints = {}
    pbar = tqdm(range(iters), desc="PGD Attack (Full Channel)")

    # 2. Precompute normalized frequency mask
    if frequency_weight is not None:
        frequency_weight = frequency_weight.to(device=device, dtype=Y_q.dtype)
        norm_weight = frequency_weight / frequency_weight.max().clamp(min=1e-8)
        weighted_eps = eps * norm_weight
    else:
        weighted_eps = eps

    for i in pbar:
        actual_alpha = alpha - (alpha - alpha / 100) * (i / iters)

        # 3. Apply perturbations
        Y_perturbed = Y_q + delta_Y
        Cb_perturbed = Cb_q + delta_Cb
        Cr_perturbed = Cr_q + delta_Cr

        # 4. Differentiable JPEG Decode
        # Set ste=True so that gradients can pass through the rounding steps
        # defined in your utils.zip/jpeg_dct.py
        X_adv = jpeg_decode(
            y_encoded=Y_perturbed,
            cb_encoded=Cb_perturbed,
            cr_encoded=Cr_perturbed,
            H=H,
            W=W,
            jpeg_quality=jpeg_q,
            ste=True,  # Critical for gradient flow
        )

        # Normalize for VAE (Standard Stable Diffusion VAE expects [-1, 1])
        X_adv = (X_adv / 255.0) * 2.0 - 1.0
        X_adv = X_adv.to(dtype=vae_encoder.dtype)

        # 5. VAE Forward & Loss
        latent_dist = vae_encoder.encode(X_adv).latent_dist
        loss = latent_dist.mean.norm()

        pbar.set_description(f"Loss: {loss.item():.6f}")

        # 6. Compute Gradients for all deltas
        grads = torch.autograd.grad(loss, [delta_Y, delta_Cb, delta_Cr])
        grad_Y, grad_Cb, grad_Cr = grads[0], grads[1], grads[2]

        with torch.no_grad():
            # Update and Clip Y
            delta_Y = torch.clamp(
                delta_Y - actual_alpha * grad_Y.sign(), -weighted_eps, weighted_eps
            )
            # Update and Clip Cb
            delta_Cb = torch.clamp(
                delta_Cb - actual_alpha * grad_Cb.sign(), -weighted_eps, weighted_eps
            )
            # Update and Clip Cr
            delta_Cr = torch.clamp(
                delta_Cr - actual_alpha * grad_Cr.sign(), -weighted_eps, weighted_eps
            )

        # Re-enable gradients for next iteration
        delta_Y.requires_grad_(True)
        delta_Cb.requires_grad_(True)
        delta_Cr.requires_grad_(True)

        # Checkpoint saving (saving all three channels)
        if checkpoints is not None and (i + 1) in checkpoints:
            saved_checkpoints[i + 1] = {
                "Y": (Y_q + delta_Y).detach().cpu(),
                "Cb": (Cb_q + delta_Cb).detach().cpu(),
                "Cr": (Cr_q + delta_Cr).detach().cpu(),
            }

    # Final results
    Y_final = (Y_q + delta_Y).detach()
    Cb_final = (Cb_q + delta_Cb).detach()
    Cr_final = (Cr_q + delta_Cr).detach()

    if checkpoints is not None and saved_checkpoints:
        save_path = (
            base_checkpoint_path / str(image_id) / f"dct_checkpoints_{image_id}.pt"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved_checkpoints, save_path)

    return Y_final, Cb_final, Cr_final, loss


def generate_pgd_dct_protection(
    iters: int,
    eps: float,
    alpha: float,
    image_paths: list[Path],
    jpeg_quality: int,
    vae_encoder: Any,
    frequency_weight: torch.Tensor,
):
    from utils.data_controller import load_image_tensor
    from utils.optimizer import pgd_dct
    from utils.jpeg_dct import jpeg_encode

    for image_path in image_paths:
        image_tensor = load_image_tensor(image_path, size=(512, 512))
        Y_q, Cb_q, Cr_q = jpeg_encode(image_tensor, jpeg_quality=jpeg_quality)

        image_id = image_path.stem.replace("_orig", "")

        Y_q_p, Cb_q_p, Cr_q_p, loss = pgd_dct(
            Y_q,
            Cb_q,
            Cr_q,
            eps=eps,
            alpha=alpha,
            vae_encoder=vae_encoder,
            jpeg_q=jpeg_quality,
            iters=iters,
            image_id=image_id,
            frequency_weight=frequency_weight,
        )

        print("Protected:", image_id, "\n")

        yield image_id, Y_q_p, Cb_q_p, Cr_q_p, loss


def wscr_dct(
    target_img: Image.Image,
    vae_encoder: Any,
    scales: list[float],
    H: int = 512,
    W: int = 512,
    jpeg_q: int = 95,
    eps: float = 1.0,
    alpha: float = 0.1,
    iters: int = 1000,
    decay: float = 1.0,  # [NEW] Momentum decay factor
    image_id: str = "unknown",
    frequency_weight: torch.Tensor = None,
    checkpoints: list[int] = None,
    checkpoint_save_path: str | Path = "results/checkpoints",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:

    device = vae_encoder.device
    saved_checkpoints = {}
    base_checkpoint_path = Path(checkpoint_save_path)
    iters_per_scale = [125, 125, 750]
    reset_points = [
        50 + 150,
        100 + 150,
        150 + 150,
        200 + 150,
        250 + 150,
        400 + 100,
        500 + 100,
        600 + 100,
        700 + 100,
        800 + 100,
    ]

    last_initiliazation_iter = sum(iters_per_scale[:-1])

    delta_Y = None

    Y_q, Cb_q, Cr_q = None, None, None

    global_iter = 0

    loss = None

    for scale_idx, scale in enumerate(scales):

        size = (int(H * scale), int(W * scale))

        if (
            delta_Y is not None
            and Y_q is not None
            and Cb_q is not None
            and Cr_q is not None
        ):
            prev_scale_y_perturbed = Y_q + delta_Y

            orig_scaled_img_tensor = load_image_tensor(target_img, size=size)
            Y_q_o, Cb_q_o, Cr_q_o = jpeg_encode(orig_scaled_img_tensor, jpeg_q)

            X_adv = jpeg_decode(
                y_encoded=prev_scale_y_perturbed,
                cb_encoded=Cb_q,
                cr_encoded=Cr_q,
                H=int(H * scales[scale_idx - 1]),
                W=int(W * scales[scale_idx - 1]),
                jpeg_quality=jpeg_q,
            )

            prev_x_adv_tensor = resize_image_tensor(X_adv, size=size)
            Y_q_p, Cb_q_p, Cr_q_p = jpeg_encode(prev_x_adv_tensor, jpeg_q)

            # 1. Update the base variables to the new scale so shapes match in the loop
            Y_q = Y_q_o
            Cb_q = Cb_q_o
            Cr_q = Cr_q_o

            # 2. Delta should be (Perturbed Upscaled - Original Upscaled)
            delta_Y = Y_q_p - Y_q_o
        else:
            image_tensor = load_image_tensor(target_img, size=size)
            Y_q, Cb_q, Cr_q = jpeg_encode(image_tensor, jpeg_q)
            delta_Y = torch.zeros_like(Y_q, device=device)

        # Initialize perturbation

        # [NEW] Initialize Momentum Buffer
        momentum = torch.zeros_like(Y_q, device=device)
        # [NEW] Define minimum alpha for the schedule (e.g., 1% of initial alpha)
        alpha_min = alpha * 0.01
        scale_iters = iters_per_scale[scale_idx]

        pbar = tqdm(
            range(scale_iters),
            desc="Warm start Cold reset [WSCR - prototype] DCT Attack",
        )

        # [NEW] Initialize tracking variables for the current scale
        best_loss = float("inf")
        best_delta_Y = delta_Y.detach().clone()

        # Precompute normalized frequency mask
        if frequency_weight is not None:
            frequency_weight = frequency_weight.to(device=device, dtype=delta_Y.dtype)
            norm_weight = frequency_weight / frequency_weight.max().clamp(min=1e-8)
            weighted_eps = eps * norm_weight
        else:
            weighted_eps = None

        # [NEW] Track the last time we reset to calculate the Cosine period properly.
        # Starting a new scale acts as our first "reset" point for the schedule.
        last_reset_iter = global_iter

        for i in pbar:

            # ── 1. COLD RESET LOGIC ──
            if (
                global_iter - 1 in reset_points
                and global_iter > last_initiliazation_iter
            ):
                # In-place zeroing of the momentum tensor (fast and memory efficient)
                momentum.zero_()
                # Mark this as the start of a new learning rate cycle
                last_reset_iter = global_iter

                # [NEW] Restore delta_Y to the best found so far
                delta_Y = best_delta_Y.detach().clone()

                print("Cold Reset at:", global_iter)

            # ── 2. DYNAMIC COSINE ANNEALING (Warm Restarts) ──
            # Calculate how many steps we've taken in the current cycle
            steps_since_reset = global_iter - last_reset_iter

            # Calculate the length of the current cycle (until next reset OR end of scale)
            end_of_scale_global_iter = (global_iter - i) + scale_iters
            next_reset = next(
                (
                    rp
                    for rp in reset_points
                    if (rp > last_reset_iter and rp > last_initiliazation_iter)
                ),
                None,
            )

            if next_reset is not None and next_reset < end_of_scale_global_iter:
                cycle_length = next_reset - last_reset_iter
            else:
                cycle_length = end_of_scale_global_iter - last_reset_iter

            # Prevent division by zero just in case
            cycle_length = max(1, cycle_length)

            # Decays from 'alpha' to 'alpha_min' following a cosine curve for the CURRENT cycle
            actual_alpha = alpha_min + 0.5 * (alpha - alpha_min) * (
                1 + math.cos((steps_since_reset / cycle_length) * math.pi)
            )

            delta_Y.requires_grad_(True)

            # Apply perturbation
            Y_q_perturbed = Y_q + delta_Y

            # DiffJPEG decode
            X_adv = jpeg_decode(
                y_encoded=Y_q_perturbed,
                cb_encoded=Cb_q,
                cr_encoded=Cr_q,
                H=size[0],
                W=size[1],
                jpeg_quality=jpeg_q,
            )

            X_adv = (X_adv / 255.0) * 2.0 - 1.0
            X_adv = X_adv.to(dtype=vae_encoder.dtype)

            # VAE forward + loss
            latent_dist = vae_encoder.encode(X_adv).latent_dist
            loss = latent_dist.mean.norm()

            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_delta_Y = delta_Y.detach().clone()

            pbar.set_description(
                f"Loss: {loss.item():.4f} | Alpha: {actual_alpha:.5f} | best loss: {best_loss:.5f}"
            )

            # Gradient calculation
            grad = torch.autograd.grad(loss, delta_Y, only_inputs=True)[0]

            with torch.no_grad():
                # ── Momentum Step ──
                reduction_dims = tuple(range(1, grad.ndim)) if grad.ndim > 1 else (0,)
                grad_norm = torch.mean(
                    torch.abs(grad), dim=reduction_dims, keepdim=True
                )

                normalized_grad = grad / (grad_norm + 1e-12)

                # Update Momentum
                momentum = momentum * decay + normalized_grad

                # Update Delta using Momentum and the cycling actual_alpha
                delta_Y = delta_Y - actual_alpha * momentum.sign()

                # ── Weighted / Adaptive Clipping ──
                if weighted_eps is not None:
                    delta_Y = torch.maximum(
                        torch.minimum(delta_Y, weighted_eps), -weighted_eps
                    )
                else:
                    delta_Y = torch.clamp(delta_Y, -eps, eps)

            delta_Y = delta_Y.detach()

            # Checkpoint saving
            if checkpoints is not None and (i + 1) in checkpoints:
                Y_q_int = (Y_q + best_delta_Y).detach().cpu()
                saved_checkpoints[i + 1] = Y_q_int

            global_iter += 1

        Y_q_final = (Y_q + delta_Y).detach()

        if checkpoints is not None and saved_checkpoints:
            save_path = (
                base_checkpoint_path / str(image_id) / f"y_q_checkpoints_{image_id}.pt"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                saved_checkpoints,
                save_path,
                _use_new_zipfile_serialization=True,
            )

    torch.cuda.empty_cache()

    return Y_q_final, Cb_q, Cr_q, loss


def wscr_dct_full(
    target_img: Image.Image,
    vae_encoder: Any,
    scales: list[float],
    H: int = 512,
    W: int = 512,
    jpeg_q: int = 95,
    eps: float = 1.0,
    alpha: float = 0.1,
    iters: int = 1000,
    decay: float = 1.0,  # Momentum decay factor
    image_id: str = "unknown",
    frequency_weight: torch.Tensor = None,
    checkpoints: list[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, dict]:

    device = vae_encoder.device
    saved_checkpoints = {}

    iters_per_scale = [
        100,
        100,
        800,
    ]

    reset_points = [100, 200, 400, 800]

    last_initiliazation_iter = sum(iters_per_scale[:-1])

    adjusted_checkpoints = checkpoints

    # Initialize delta tracking for all 3 channels
    delta_Y, delta_Cb, delta_Cr = None, None, None
    Y_q, Cb_q, Cr_q = None, None, None

    global_iter = 0
    loss = None

    # Start timer here to avoid NameError
    t0 = time.time()
    dt = None
    mem_mib = None

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=device)

    for scale_idx, scale in enumerate(scales):

        size = (int(H * scale), int(W * scale))

        if (
            delta_Y is not None
            and delta_Cb is not None
            and delta_Cr is not None
            and Y_q is not None
            and Cb_q is not None
            and Cr_q is not None
        ):
            # Apply perturbation to all channels for warm start decode
            prev_scale_y_perturbed = Y_q + delta_Y
            prev_scale_cb_perturbed = Cb_q + delta_Cb
            prev_scale_cr_perturbed = Cr_q + delta_Cr

            orig_scaled_img_tensor = load_image_tensor(target_img, size=size)
            Y_q_o, Cb_q_o, Cr_q_o = jpeg_encode(orig_scaled_img_tensor, jpeg_q)

            X_adv = jpeg_decode(
                y_encoded=prev_scale_y_perturbed,
                cb_encoded=prev_scale_cb_perturbed,
                cr_encoded=prev_scale_cr_perturbed,
                H=int(H * scales[scale_idx - 1]),
                W=int(W * scales[scale_idx - 1]),
                jpeg_quality=jpeg_q,
            )

            prev_x_adv_tensor = resize_image_tensor(X_adv, size=size)
            Y_q_p, Cb_q_p, Cr_q_p = jpeg_encode(prev_x_adv_tensor, jpeg_q)

            # 1. Update the base variables to the new scale so shapes match in the loop
            Y_q = Y_q_o
            Cb_q = Cb_q_o
            Cr_q = Cr_q_o

            # 2. Delta should be (Perturbed Upscaled - Original Upscaled)
            delta_Y = Y_q_p - Y_q_o
            delta_Cb = Cb_q_p - Cb_q_o
            delta_Cr = Cr_q_p - Cr_q_o
        else:
            image_tensor = load_image_tensor(target_img, size=size)
            Y_q, Cb_q, Cr_q = jpeg_encode(image_tensor, jpeg_q)
            delta_Y = torch.zeros_like(Y_q, device=device)
            delta_Cb = torch.zeros_like(Cb_q, device=device)
            delta_Cr = torch.zeros_like(Cr_q, device=device)

        # Initialize Momentum Buffers for all channels
        momentum_Y = torch.zeros_like(Y_q, device=device)
        momentum_Cb = torch.zeros_like(Cb_q, device=device)
        momentum_Cr = torch.zeros_like(Cr_q, device=device)

        alpha_min = alpha * 0.01
        scale_iters = iters_per_scale[scale_idx]

        pbar = tqdm(
            range(scale_iters),
            desc="Warm start Cold reset [WSCR - prototype] DCT Attack",
        )

        # Precompute normalized frequency mask
        if frequency_weight is not None:
            frequency_weight = frequency_weight.to(device=device, dtype=delta_Y.dtype)
            norm_weight = frequency_weight / frequency_weight.max().clamp(min=1e-8)
            weighted_eps = eps * norm_weight
        else:
            weighted_eps = None

        last_reset_iter = global_iter

        for i in pbar:

            # ── 1. COLD RESET LOGIC ──
            if (
                global_iter - 1 in reset_points
                and global_iter > last_initiliazation_iter
            ):
                # Zero momentum for all channels (Soft Reset)
                momentum_Y.zero_()
                momentum_Cb.zero_()
                momentum_Cr.zero_()

                last_reset_iter = global_iter
                print("Cold Reset at:", global_iter)

            # ── 2. DYNAMIC COSINE ANNEALING (Warm Restarts) ──
            steps_since_reset = global_iter - last_reset_iter
            end_of_scale_global_iter = (global_iter - i) + scale_iters

            next_reset = next(
                (
                    rp
                    for rp in reset_points
                    if (rp > last_reset_iter and rp > last_initiliazation_iter)
                ),
                None,
            )

            if next_reset is not None and next_reset < end_of_scale_global_iter:
                cycle_length = next_reset - last_reset_iter
            else:
                cycle_length = end_of_scale_global_iter - last_reset_iter

            cycle_length = max(1, cycle_length)

            actual_alpha = alpha_min + 0.5 * (alpha - alpha_min) * (
                1 + math.cos((steps_since_reset / cycle_length) * math.pi)
            )

            # Require grad on all channels
            delta_Y.requires_grad_(True)
            delta_Cb.requires_grad_(True)
            delta_Cr.requires_grad_(True)

            # Apply perturbation to all channels
            Y_q_perturbed = Y_q + delta_Y
            Cb_q_perturbed = Cb_q + delta_Cb
            Cr_q_perturbed = Cr_q + delta_Cr

            # DiffJPEG decode with all perturbed channels
            X_adv = jpeg_decode(
                y_encoded=Y_q_perturbed,
                cb_encoded=Cb_q_perturbed,
                cr_encoded=Cr_q_perturbed,
                H=size[0],
                W=size[1],
                jpeg_quality=jpeg_q,
            )

            X_adv = (X_adv / 255.0) * 2.0 - 1.0
            X_adv = X_adv.to(dtype=vae_encoder.dtype)

            # VAE forward + loss
            latent_dist = vae_encoder.encode(X_adv).latent_dist
            loss = latent_dist.mean.norm()

            # (Removed best_loss / best_delta tracking logic here)

            pbar.set_description(f"Loss: {loss.item():.4f} | Alpha: {actual_alpha:.5f}")

            # Gradient calculation for all channels simultaneously
            grad_Y, grad_Cb, grad_Cr = torch.autograd.grad(
                loss, (delta_Y, delta_Cb, delta_Cr), only_inputs=True
            )

            with torch.no_grad():
                # ── Momentum Step for Y ──
                red_dims_Y = tuple(range(1, grad_Y.ndim)) if grad_Y.ndim > 1 else (0,)
                grad_norm_Y = torch.mean(
                    torch.abs(grad_Y), dim=red_dims_Y, keepdim=True
                )
                normalized_grad_Y = grad_Y / (grad_norm_Y + 1e-12)
                momentum_Y = momentum_Y * decay + normalized_grad_Y
                delta_Y = delta_Y - actual_alpha * momentum_Y.sign()

                # ── Momentum Step for Cb ──
                red_dims_Cb = (
                    tuple(range(1, grad_Cb.ndim)) if grad_Cb.ndim > 1 else (0,)
                )
                grad_norm_Cb = torch.mean(
                    torch.abs(grad_Cb), dim=red_dims_Cb, keepdim=True
                )
                normalized_grad_Cb = grad_Cb / (grad_norm_Cb + 1e-12)
                momentum_Cb = momentum_Cb * decay + normalized_grad_Cb
                delta_Cb = delta_Cb - actual_alpha * momentum_Cb.sign()

                # ── Momentum Step for Cr ──
                red_dims_Cr = (
                    tuple(range(1, grad_Cr.ndim)) if grad_Cr.ndim > 1 else (0,)
                )
                grad_norm_Cr = torch.mean(
                    torch.abs(grad_Cr), dim=red_dims_Cr, keepdim=True
                )
                normalized_grad_Cr = grad_Cr / (grad_norm_Cr + 1e-12)
                momentum_Cr = momentum_Cr * decay + normalized_grad_Cr
                delta_Cr = delta_Cr - actual_alpha * momentum_Cr.sign()

                # ── Weighted / Adaptive Clipping ──
                if weighted_eps is not None:
                    # Apply clipping to all channels
                    delta_Y = torch.maximum(
                        torch.minimum(delta_Y, weighted_eps), -weighted_eps
                    )
                    delta_Cb = torch.maximum(
                        torch.minimum(delta_Cb, weighted_eps), -weighted_eps
                    )
                    delta_Cr = torch.maximum(
                        torch.minimum(delta_Cr, weighted_eps), -weighted_eps
                    )
                else:
                    delta_Y = torch.clamp(delta_Y, -eps, eps)
                    delta_Cb = torch.clamp(delta_Cb, -eps, eps)
                    delta_Cr = torch.clamp(delta_Cr, -eps, eps)

            # Detach all
            delta_Y = delta_Y.detach()
            delta_Cb = delta_Cb.detach()
            delta_Cr = delta_Cr.detach()

            # Checkpoint saving (Saves the CURRENT delta now)
            if checkpoints is not None and (i + 1) in adjusted_checkpoints:
                idx = adjusted_checkpoints.index(i + 1)
                key = checkpoints[idx]
                saved_checkpoints[key] = {
                    "Y_q": (Y_q + delta_Y).detach().cpu(),
                    "Cb_q": (Cb_q + delta_Cb).detach().cpu(),
                    "Cr_q": (Cr_q + delta_Cr).detach().cpu(),
                }

            global_iter += 1

        dt = time.time() - t0

        final_peak_bytes = torch.cuda.max_memory_allocated(device=device)
        mem_mib = final_peak_bytes / (1024**2)

        # Compute final perturbed channels
        Y_q_final = (Y_q + delta_Y).detach()
        Cb_q_final = (Cb_q + delta_Cb).detach()
        Cr_q_final = (Cr_q + delta_Cr).detach()

    # Detach loss to prevent graph leak
    if loss is not None and isinstance(loss, torch.Tensor):
        loss = loss.item()

    torch.cuda.empty_cache()

    # Return all final perturbed channels
    return Y_q_final, Cb_q_final, Cr_q_final, loss, dt, mem_mib, saved_checkpoints


def generate_wscr_dct_protection(
    iters: int,
    eps: float,
    alpha: float,
    image_paths: list[Path],
    jpeg_quality: int,
    vae_encoder: Any,
    checkpoints: list[int] = None,
    frequency_weight: torch.Tensor = None,
):
    H, W = 512, 512

    for image_path in image_paths:

        target_img = Image.open(image_path).resize((H, W)).convert("RGB")

        image_id = image_path.stem.replace("_orig", "")

        Y_q_p, Cb_q_p, Cr_q_p, loss, dt, mem_mib, saved_checkpoints = wscr_dct_full(
            target_img=target_img,
            vae_encoder=vae_encoder,
            scales=[1 / 4, 1 / 2, 1],
            H=H,
            W=W,
            eps=eps,
            alpha=alpha,
            jpeg_q=jpeg_quality,
            iters=iters,
            image_id=image_id,
            frequency_weight=frequency_weight,
            checkpoints=checkpoints,
        )

        print(
            f"Protected: {image_id} | Time: {dt:.2f}s | Peak Mem: {mem_mib:.2f} MiB\n"
        )

        yield image_id, Y_q_p, Cb_q_p, Cr_q_p, loss, dt, mem_mib, saved_checkpoints
