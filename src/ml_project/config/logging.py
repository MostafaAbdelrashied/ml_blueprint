import sys
from typing import Optional

from loguru import logger


def setup_logging(level: Optional[str] = None) -> None:
    """Configure logging settings for the entire application.

    Args:
        level: The logging level to use. Defaults to "INFO" if not specified.
               Valid values are "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    """
    # Remove any existing handlers
    logger.remove()

    # Set default level if none provided
    if level is None:
        level = "INFO"

    # Validate logging level
    valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    level = level.upper()
    if level not in valid_levels:
        raise ValueError(f"Invalid logging level. Must be one of {valid_levels}")

    # Add new handler with specified level
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        compression="zip",
    )
