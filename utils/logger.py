from __future__ import annotations
"""
Structured logging utility for the Sector War Graph project.
Provides colored, timestamped logging with module-level control.
"""

import logging
import sys
from datetime import datetime


class ColorFormatter(logging.Formatter):
    """Custom formatter with color support for terminal output."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a structured logger for a given module.
    
    Args:
        name: Module name (e.g., 'data.acled_pipeline')
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f'sector_war_graph.{name}')
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        fmt = ColorFormatter(
            '%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger


def log_step(logger: logging.Logger, step: str, detail: str = ''):
    """Log a pipeline step with visual separator."""
    logger.info(f'{"─" * 40}')
    logger.info(f'▶ {step}')
    if detail:
        logger.info(f'  {detail}')


def log_dataframe_info(logger: logging.Logger, df, name: str):
    """Log DataFrame shape and column info."""
    logger.info(f'{name}: shape={df.shape}, '
                f'columns={list(df.columns)[:5]}{"..." if len(df.columns) > 5 else ""}')
