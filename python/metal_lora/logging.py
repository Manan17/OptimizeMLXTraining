"""Logging configuration for MetalLoRA."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "metal_lora", level: Optional[int] = None) -> logging.Logger:
    """Get configured logger."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
    
    if level is not None:
        logger.setLevel(level)
    elif logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    
    return logger


def set_log_level(level: int) -> None:
    """Set global log level."""
    get_logger().setLevel(level)


def enable_debug() -> None:
    """Enable debug logging."""
    set_log_level(logging.DEBUG)


def disable_logging() -> None:
    """Disable all logging."""
    set_log_level(logging.CRITICAL + 1)


# Default logger
logger = get_logger()
