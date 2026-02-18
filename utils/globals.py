import torch

from datetime import datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_timestamp_based_name(
    prefix: str, suffix: str = "", date_only: bool = False
) -> str:

    if date_only:
        timestamp = datetime.now().strftime("%m-%d-%Y")
        return prefix + timestamp + suffix

    timestamp = datetime.now().strftime("%m-%d-%Y_at_%I-%M-%S_%p")
    return prefix + timestamp + suffix
