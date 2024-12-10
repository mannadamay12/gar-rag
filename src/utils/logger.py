import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO):
    """Configure logger with console and file handlers"""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create logger with name
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file provided
    if log_file:
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_path / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger