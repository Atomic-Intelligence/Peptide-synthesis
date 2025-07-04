import sys
import warnings
from pathlib import Path

from loguru import logger


def setup_logger():
    warnings.filterwarnings("ignore")
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Remove default handler
    logger.remove()

    # Beautiful console output with colors and emojis
    console_format = (
        "<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> | "
        "<level>{level: <8}</level> | "
        "<level>{message: <80}</level> "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
    )

    # Detailed file output
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "process:{process} | "
        "{name}:{function}:{line} | "
        "{message} | "
    )

    # Add handlers
    logger.add(
        sys.stderr,
        format=console_format,
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )

    # Separate log files for different levels
    logger.add(
        log_dir / "info.log",
        format=file_format,
        level="INFO",
        rotation="10 MB",
        retention="1 month",
        compression="zip",
        encoding="utf8",
        enqueue=True,
    )

    logger.add(
        log_dir / "errors.log",
        format=file_format,
        level="ERROR",
        rotation="10 MB",
        retention="1 month",
        compression="zip",
        encoding="utf8",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )


    return logger