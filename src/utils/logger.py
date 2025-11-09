"""Structured logging setup using loguru."""

import sys
from pathlib import Path
from loguru import logger

# Remove default handler
logger.remove()

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Console handler - colorized output for development
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
    backtrace=True,
    diagnose=True
)

# File handler - detailed logs with rotation
logger.add(
    "logs/llm_dashboard.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB",  # Rotate when file reaches 10 MB
    retention="30 days",  # Keep logs for 30 days
    compression="zip",  # Compress rotated logs
    backtrace=True,
    diagnose=True,
    enqueue=True  # Async logging for better performance
)

# Error-only file handler - separate file for errors
logger.add(
    "logs/errors.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",
    rotation="5 MB",
    retention="60 days",  # Keep error logs longer
    compression="zip",
    backtrace=True,
    diagnose=True,
    enqueue=True
)

# Export logger
__all__ = ["logger"]


def set_log_level(level: str) -> None:
    """
    Set logging level dynamically.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger.remove()  # Remove all handlers

    # Re-add console handler with new level
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level.upper(),
        colorize=True,
        backtrace=True,
        diagnose=True
    )

    # Re-add file handlers
    logger.add(
        "logs/llm_dashboard.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",  # Keep file logging at DEBUG
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
        enqueue=True
    )

    logger.add(
        "logs/errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="5 MB",
        retention="60 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
        enqueue=True
    )

    logger.info(f"Log level set to {level.upper()}")


def get_logger(name: str = None):
    """
    Get a logger instance with optional name binding.

    Args:
        name: Optional name to bind to logger context

    Returns:
        Logger instance

    Example:
        >>> from src.utils.logger import get_logger
        >>> log = get_logger(__name__)
        >>> log.info("This is a test")
    """
    if name:
        return logger.bind(name=name)
    return logger
