# src/router/logger_utils.py
"""Centralized logging utilities for router training and evaluation."""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_router_logger(
    log_dir: str | Path,
    name: str = "router",
    level: int = logging.INFO,
    console_level: int = logging.INFO,
) -> tuple[logging.Logger, Path]:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        log_dir: Directory to store log files
        name: Logger name (used for log file prefix)
        level: File logging level
        console_level: Console logging level
    
    Returns:
        (logger, log_file_path)
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Remove any existing handlers
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10_000_000,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized: {log_file}")
    logger.info("=" * 80)
    
    return logger, log_file


def log_config(logger: logging.Logger, config: dict, title: str = "Configuration"):
    """Log configuration dictionary in a readable format."""
    logger.info(f"{title}:")
    logger.info("-" * 80)
    for key, value in sorted(config.items()):
        logger.info(f"  {key:25s} = {value}")
    logger.info("-" * 80)


def log_model_summary(logger: logging.Logger, model, title: str = "Model Summary"):
    """Log model architecture and parameter count."""
    try:
        import torch.nn as nn
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"{title}:")
        logger.info("-" * 80)
        logger.info(f"  Total parameters:      {total_params:,}")
        logger.info(f"  Trainable parameters:  {trainable_params:,}")
        logger.info(f"  Model architecture:")
        for line in str(model).split('\n'):
            logger.info(f"    {line}")
        logger.info("-" * 80)
    except Exception as e:
        logger.warning(f"Could not log model summary: {e}")


def log_epoch_metrics(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    train_metrics: dict,
    val_metrics: dict,
):
    """Log training and validation metrics for an epoch."""
    logger.info(f"Epoch [{epoch:3d}/{total_epochs:3d}]")
    logger.info(f"  Train: loss={train_metrics['loss']:.4f} | "
                f"acc(no-ties)={train_metrics['acc_no_ties']:.3f} | "
                f"acc(ties=0.5)={train_metrics['acc_ties_half']:.3f} | "
                f"+1={train_metrics['pos']:5d} -1={train_metrics['neg']:5d} 0={train_metrics['ties']:4d}")
    logger.info(f"  Val:   loss={val_metrics['loss']:.4f} | "
                f"acc(no-ties)={val_metrics['acc_no_ties']:.3f} | "
                f"acc(ties=0.5)={val_metrics['acc_ties_half']:.3f} | "
                f"+1={val_metrics['pos']:5d} -1={val_metrics['neg']:5d} 0={val_metrics['ties']:4d}")


def log_final_results(logger: logging.Logger, results: dict, title: str = "Final Results"):
    """Log final evaluation results."""
    logger.info("=" * 80)
    logger.info(f"{title}:")
    logger.info("-" * 80)
    for key, value in sorted(results.items()):
        if isinstance(value, (int, float)):
            logger.info(f"  {key:30s} = {value}")
        else:
            logger.info(f"  {key:30s} = {value}")
    logger.info("=" * 80)

