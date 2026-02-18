import torch
import csv
import yaml
import shutil
import os
import cloudinary
import cloudinary.uploader
import cloudinary.api
import time
import requests
import zipfile
import gspread

import pandas as pd

from typing import Tuple, List
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision import transforms as T
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, asdict, fields

from .globals import DEVICE, generate_timestamp_based_name


STATE_TRACKER_FILEPATH = Path("results/state_tracker.yaml")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".ico",
    ".heic",
}


@dataclass
class ExperimentResult:
    img_id: str
    frequency_weighting: str = "uniform"
    lpips: float = None
    ssim: float = None
    psnr: float = None
    vifp: float = None
    compression_lpips: float = None
    compression_lpips_degradation: float = None
    purification_lpips: float = None
    purification_lpips_degradation: float = None
    iters_10_percent_lpips: float = None
    iters_50_percent_lpips: float = None
    iters_70_percent_lpips: float = None
    iters_10_percent_ssim: float = None
    iters_50_percent_ssim: float = None
    iters_70_percent_ssim: float = None
    iters_10_percent_psnr: float = None
    iters_50_percent_psnr: float = None
    iters_70_percent_psnr: float = None
    iters_10_percent_vifp: float = None
    iters_50_percent_vifp: float = None
    iters_70_percent_vifp: float = None
    overall_runtime_sec: float = None
    peak_gpu_memory_mib: float = None


def save_tensor_as_image(image_tensor: torch.Tensor, filename: str) -> Image:
    img_tensor = image_tensor[0]
    img_tensor = torch.clamp(img_tensor, 0, 255) / 255.0
    save_image(img_tensor, filename)
    print(f"Saved image to {filename}")


def load_image_tensor(
    image: str | Path | Image.Image, size: Tuple[int, int] = None, device=DEVICE
) -> torch.Tensor:

    if isinstance(image, (str, Path)):
        print(f"Opening image from path: {image}")
        opened_image = Image.open(image).convert("RGB")
    else:
        print("Using PIL image...")
        opened_image = image

    if size is not None:
        w, h = size
    else:
        w, h = opened_image.size

    # Resize to integer multiple of 32
    # This prevents shape mismatch errors in the VAE/UNet
    w, h = map(lambda x: x - x % 32, (w, h))
    opened_image = opened_image.resize((w, h), resample=Image.LANCZOS)

    # Convert to Tensor in [0, 255] for Diff-JPEG compatibility
    image_tensor = T.ToTensor()(opened_image)  # [0, 1]
    image_tensor = image_tensor * 255.0  # [0, 255]

    # Add batch dimension [1, 3, H, W]
    return image_tensor.unsqueeze(0).to(device)


def resize_image_tensor(
    image_tensor: torch.Tensor, size: Tuple[int, int] = None
) -> torch.Tensor:
    # 1. Get current dimensions (PyTorch tensors are [Batch, Channels, Height, Width])
    _, _, current_h, current_w = image_tensor.shape

    # 2. Extract target width and height
    if size is not None:
        w, h = size
    else:
        w, h = current_w, current_h

    # 3. Resize to integer multiple of 32 (Prevents VAE/UNet mismatch)
    w = w - (w % 32)
    h = h - (h % 32)

    # 4. Skip if the tensor is already the exact right size
    if h == current_h and w == current_w:
        return image_tensor

    # 5. GPU-Native Resize
    # - mode='bicubic' with antialias=True is PyTorch's closest equivalent to PIL's LANCZOS
    # - F.interpolate expects the size argument as (height, width)
    resized_tensor = F.interpolate(
        image_tensor, size=(h, w), mode="bicubic", align_corners=False, antialias=True
    )

    # 6. Clamp to [0, 255]
    # Bicubic interpolation can slightly overshoot/undershoot boundaries, so we clamp it back.
    resized_tensor = resized_tensor.clamp(0.0, 255.0)

    return resized_tensor


def load_saved_checkpoint(filename: str, device=DEVICE) -> torch.Tensor:
    saved_file_path = Path(filename)

    saved_y_q_perturbed = torch.load(
        saved_file_path,
    ).to(device)

    return saved_y_q_perturbed


def get_filepath_batch(
    directory: str | Path,
    batch_size: int,
    current_filename: str | None = None,
    extensions: tuple[str, ...] | None = None,
    exclude_current: bool = False,
) -> list[Path]:
    directory = Path(directory)

    files = [
        p for p in directory.iterdir() if extensions is None or p.suffix in extensions
    ]

    files.sort()

    if current_filename is None:
        start_idx = 0
    else:
        try:
            start_idx = next(
                i for i, f in enumerate(files) if f.name == current_filename
            ) + (1 if exclude_current else 0)
        except StopIteration:
            raise ValueError(
                f"Filename '{current_filename}' not found in directory '{directory}'"
            )

    end_idx = start_idx + batch_size
    return files[start_idx:end_idx]


def load_dataset(
    metadata_df: pd.DataFrame,
    batch_size: int,
    start_id: str,
    orig_dir: Path,
    edited_dir: Path,
    exclude_current: bool = False,
    no_cache: bool = False,
) -> tuple[list[str], list[str], list[str], list[str]]:

    batch_size = batch_size + 1 if exclude_current else batch_size

    mask = metadata_df["img_id"] >= start_id
    batch_metadata = (
        metadata_df[mask]
        .sort_values("img_id")
        .head(batch_size)[["img_id", "edit_prompt", "orig_img_url", "edited_img_url"]]
    )

    orig_dir.mkdir(parents=True, exist_ok=True)
    edited_dir.mkdir(parents=True, exist_ok=True)

    img_id_list = []
    img_src_orig_paths = []
    img_src_edited_paths = []
    edit_prompts = []

    for i, row in batch_metadata.iterrows():
        img_id = row["img_id"]
        orig_url = row["orig_img_url"]
        edited_url = row["edited_img_url"]

        orig_out = orig_dir / f"{img_id}_orig.png"
        edited_out = edited_dir / f"{img_id}_edited.png"

        if not orig_out.exists() or no_cache:
            download_png_lossless(url=orig_url, out_path=orig_out)
        if not edited_out.exists() or no_cache:
            download_png_lossless(url=edited_url, out_path=edited_out)

        if not (exclude_current and img_id == start_id):
            img_id_list.append(img_id)
            img_src_orig_paths.append(orig_out)
            img_src_edited_paths.append(edited_out)
            edit_prompts.append(row["edit_prompt"])

    return img_id_list, edit_prompts, img_src_orig_paths, img_src_edited_paths


def loadstate_last_processed_image() -> str:
    if not STATE_TRACKER_FILEPATH.exists():
        raise FileNotFoundError("State tracker file not found")

    with open(STATE_TRACKER_FILEPATH, "r") as f:
        state = yaml.safe_load(f)

    return state["LAST_PROCESSED_IMAGE_ID"]


def savestate_last_processed_image(image_id: str) -> None:
    STATE_TRACKER_FILEPATH.parent.mkdir(parents=True, exist_ok=True)

    with open(STATE_TRACKER_FILEPATH, "w") as f:
        yaml.safe_dump({"LAST_PROCESSED_IMAGE_ID": image_id}, f)


def check_paths_exist(paths_dict: dict, strict: bool = False) -> None:
    missing = [name for name, path in paths_dict.items() if not path.exists()]

    if not missing:
        return

    if strict:
        raise FileNotFoundError(
            "Missing required dataset paths:\n"
            + "\n".join(f"- {name}: {paths_dict[name]}" for name in missing)
        )

    for name in missing:
        path = paths_dict[name]
        # parents=True creates the whole tree; exist_ok=True prevents race condition errors
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created missing directory: {name} -> {path}")


def save_image_lossless(pil_image: Image.Image, output_path: str | Path):
    """
    Save a PIL Image to disk as a lossless PNG.
    Ensures the output directory exists.

    Args:
        pil_image (PIL.Image.Image): Image to save
        output_path (str | Path): Destination file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pil_image.save(
        output_path,
        format="PNG",
        compress_level=9,  # max compression, still lossless
        optimize=False,  # avoid any entropy/palette optimizations
    )


def download_png_lossless(url: str, out_path: Path, timeout=15):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def move_to_temps(
    source_path: str | Path,
    temps_dir: str | Path = "temps",
):
    source_path = Path(source_path)
    target_path = Path(temps_dir) / generate_timestamp_based_name("temp_")

    target_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(source_path, target_path)


def write_results_csv(csv_path: str | Path, result: ExperimentResult) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Convert to dict and filter: Only keep fields that are NOT None
    # This ensures we don't create columns for empty data on the first run
    row_data = {k: v for k, v in asdict(result).items() if v is not None}

    file_exists = csv_path.exists()

    if file_exists:
        # 2. If file exists, we MUST read the existing headers first.
        # We have to conform to the existing file structure.
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
    else:
        # 3. If new file, define headers based ONLY on the non-empty fields we have right now
        fieldnames = list(row_data.keys())

    # 4. Write the data
    with csv_path.open("a", newline="") as f:
        # extrasaction="ignore" is crucial here:
        # It prevents a crash if 'row_data' contains keys that aren't in 'fieldnames'
        # (e.g., if the file was created with fewer columns than you have now).
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_data)


def write_results_gsheets(
    credentials_dict: dict, sheet_name: str, csv_path: str, tab_name: str = None
) -> gspread.worksheet.JSONResponse:

    # 1. Load Credentials
    credentials_dict = {k.lower(): v for k, v in credentials_dict.items()}
    if not credentials_dict:
        raise ValueError("Key 'google_cloud' not found in secrets.yaml")

    # 2. Connect to Google Sheets
    try:
        gc = gspread.service_account_from_dict(credentials_dict)
        sh = gc.open(sheet_name)

        # Track if we need to write headers
        should_write_header = False

        if tab_name:
            try:
                worksheet = sh.worksheet(tab_name)
                # If tab exists, check if it's empty to decide on headers
                if not worksheet.get_values("A1:A1"):
                    should_write_header = True
            except gspread.WorksheetNotFound:
                print(f"Tab '{tab_name}' not found. Creating it...")
                worksheet = sh.add_worksheet(title=tab_name, rows=1000, cols=20)
                # A new tab definitely needs headers
                should_write_header = True
        else:
            worksheet = sh.sheet1
            if not worksheet.get_values("A1:A1"):
                should_write_header = True

    except Exception as e:
        print(f"Google API connection failed: {e}")
        return None

    # 3. Read CSV file
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return None

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Get headers dynamically
        csv_headers = reader.fieldnames

        if not csv_headers:
            print("CSV file is empty (no headers found)")
            return None

        rows = list(reader)

    # 4. Prepare upload data
    # Prepend 'timestamp' to the dynamic CSV headers
    cloud_columns = ["timestamp"] + csv_headers
    batch_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    upload_data = []

    # FIX: Add headers if the sheet is new OR empty (regardless of whether we have data rows)
    if should_write_header:
        upload_data.append(cloud_columns)

    # Add data rows (if any exist)
    if rows:
        for record in rows:
            row_data = []
            for col in cloud_columns:
                if col == "timestamp":
                    row_data.append(batch_timestamp)
                else:
                    row_data.append(record.get(col, ""))
            upload_data.append(row_data)

    # 5. Upload (Only if we have something to write)
    if not upload_data:
        print("No headers needed and no data rows to append.")
        return None

    try:
        response = worksheet.append_rows(upload_data, value_input_option="USER_ENTERED")
        return response
    except Exception as e:
        print(f"Failed to upload rows to Google Sheets: {e}")
        return None


def cloudinary_upload_one_file(
    local_root: Path,
    local_path: Path,
    cloudinary_folder: Path,
) -> str:
    """
    Upload a single file to Cloudinary, preserving subfolder structure.
    """
    # Make sure local_root and local_path are Path objects
    local_root = Path(local_root)
    local_path = Path(local_path)
    cloudinary_folder = Path(cloudinary_folder)

    # Relative path from the root, POSIX style (forward slashes)
    rel_dir = local_path.parent.relative_to(local_root).as_posix()

    dest_folder = (
        cloudinary_folder.as_posix()
        if rel_dir == "."
        else Path(cloudinary_folder, rel_dir).as_posix()
    )

    ext = local_path.suffix.lower()

    # check extension
    if ext in IMAGE_EXTENSIONS:
        upload_kwargs = {
            "folder": dest_folder,
            "use_filename": True,
            "unique_filename": False,
            # no resource_type => default is image
        }
    else:
        upload_kwargs = {
            "folder": dest_folder,
            "use_filename": True,
            "unique_filename": False,
            "resource_type": "auto",  # not an image
        }

    result = cloudinary.uploader.upload(str(local_path), **upload_kwargs)

    return result["url"]


def cloudinary_upload_folder(
    folder_path: str | Path,
    cloudinary_folder: str | Path,
    max_workers: int = 8,
) -> List[str]:
    """
    Recursively uploads all files in folder_path to Cloudinary in parallel.
    Returns the list of uploaded URLs.
    """
    folder_path = Path(folder_path)
    local_files: List[Path] = [p for p in folder_path.rglob("*") if p.is_file()]

    print(local_files)

    uploaded_urls: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                cloudinary_upload_one_file,
                folder_path,
                local_path,
                cloudinary_folder,
            ): local_path
            for local_path in local_files
        }

        for future in as_completed(futures):
            local_path = futures[future]
            try:
                url = future.result()
                uploaded_urls.append(url)
                print(url)
            except Exception as e:
                print(f"[UPLOAD FAILED] {local_path}: {e}")

    return uploaded_urls


def cloudinary_download_folder(
    cloudinary_folder: str, local_target_dir: str = "datasets"
):
    os.makedirs(local_target_dir, exist_ok=True)

    print(f"1. Generating download URL for: {cloudinary_folder}")
    url = cloudinary.utils.download_folder(
        cloudinary_folder,
        target_public_id="download_archive",
    )

    zip_path = os.path.join(local_target_dir, "temp_cloudinary.zip")

    print("2. Downloading ZIP...")
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        print(f"Failed to download. Status code: {r.status_code}")
        return

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)

    if os.path.getsize(zip_path) < 100:
        print("WARNING: The file is very small. It might be empty.")

    print("3. Extracting and flattening files...")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for m in zf.infolist():
                if not m.filename.startswith(cloudinary_folder):
                    continue

                name = m.filename[len(cloudinary_folder) :].lstrip("/\\")
                if not name:
                    continue

                path = os.path.join(local_target_dir, name)
                if m.is_dir():
                    os.makedirs(path, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with zf.open(m) as src, open(path, "wb") as dst:
                        shutil.copyfileobj(src, dst)

        print(f"Files extracted to: {os.path.abspath(local_target_dir)}")
    except zipfile.BadZipFile:
        print("ERROR: The downloaded file is not a valid ZIP.")
    finally:
        print("4. Cleaning up temporary ZIP file...")
        if os.path.exists(zip_path):
            os.remove(zip_path)


def cloudinary_get_asset_links(
    cloudinary_folder: str, resource_type: str = "upload", max_results: int = 500
):
    all_resources = []
    next_cursor = None

    print(f"Starting fetch for asset folder: '{cloudinary_folder}'")

    while True:
        try:
            # Fetch resources with pagination
            result = cloudinary.api.resources(
                type=resource_type,
                prefix=cloudinary_folder,
                max_results=max_results,
                next_cursor=next_cursor,
            )

            # Add current page of resources to our list
            resources = result.get("resources", [])
            all_resources.extend(resources)

            # Check if there is another page
            next_cursor = result.get("next_cursor")
            print(
                f"Fetched {len(resources)} items... (Next cursor: {'Yes' if next_cursor else 'No'})"
            )

            if not next_cursor:
                break

            # Rate limit protection
            time.sleep(0.5)

        except Exception as e:
            print(f"An error occurred: {e}")
            break

    # Extract only the secure_url strings
    secure_links = [r.get("secure_url") for r in all_resources]

    print(f"--- Complete ---")
    print(f"Total resources found: {len(all_resources)}")

    return secure_links
