import logging
import yaml
from pathlib import Path
from .globals import generate_timestamp_based_name

# Load config
# with open(config_path) as f:
#     cfg = yaml.safe_load(f)

# LOG_DIR = Path(cfg["RESULTS"]["BASE_PATH"], cfg["RESULTS"]["LOGGING"]["BASE_PATH"])
# LOG_DIR.mkdir(exist_ok=True, parents=True)

# log_filename = LOG_DIR / generate_timestamp_based_name("experiment_log_", ".log")


def init_logger(log_filename: Path) -> logging.Logger:

    print("Initializing logger:", log_filename)

    log_filename.parent.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(f"experiment_logger_{log_filename.stem}")
    logger.setLevel(logging.INFO)

    # Only add a FileHandler if there are no handlers yet
    if not logger.hasHandlers():
        fh = logging.FileHandler(log_filename)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def write_log(
    logger: logging.Logger, message: str, level: str = "info", verbose: bool = False
) -> None:
    """
    Write a log message to the given logger.
    """
    if logger is None:
        raise RuntimeError("Logger not initialized. Call init_logger() first.")

    level = level.lower()

    if verbose:
        print(f"[{level.upper()}] {message}")

    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "debug":
        logger.debug(message)
    elif level == "critical":
        logger.critical(message)
    else:
        raise ValueError(f"Unsupported log level: {level}")


def close_logger(logger: logging.Logger) -> None:
    """
    Close all handlers of the given logger.
    """

    logger.info("Closing logger...")
    if logger is None:
        return

    for handler in logger.handlers[:]:  # iterate over a copy
        handler.close()
        logger.removeHandler(handler)
