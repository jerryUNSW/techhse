"""
Logging configuration utilities.
"""

import logging
import os
from datetime import datetime
from typing import Optional

from dpprivqa.utils.config import get_config


def setup_logging(
    dataset_name: str,
    log_dir: Optional[str] = None,
    log_level: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging for an experiment.
    
    Args:
        dataset_name: Name of the dataset
        log_dir: Log directory (defaults to config)
        log_level: Log level (defaults to config)
    
    Returns:
        Configured logger
    """
    config = get_config()
    
    if log_dir is None:
        log_dir = config.get("logging", {}).get("directory", "exp-results/logs")
    
    if log_level is None:
        log_level = config.get("logging", {}).get("level", "INFO")
    
    # Create log directory for dataset
    dataset_log_dir = os.path.join(log_dir, dataset_name)
    os.makedirs(dataset_log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(dataset_log_dir, f"experiment_{timestamp}.log")
    
    # Configure logger
    logger = logging.getLogger(f"dpprivqa.{dataset_name}")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized for {dataset_name}")
    logger.info(f"Log file: {log_file}")
    
    return logger


