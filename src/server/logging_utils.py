from __future__ import annotations

import logging
import sys
from typing import Optional

from loguru import logger

_LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"


def configure_logging(level: str = "INFO") -> None:
    """Configure Loguru to replace the standard logging handlers."""
    logger.remove()
    logger.add(sys.stderr, format=_LOG_FORMAT, level=level, enqueue=True, backtrace=True, diagnose=False)
    # Bridge standard logging to Loguru so third-party modules are captured.
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin wrapper
            try:
                level_name = logger.level(record.levelname).name
            except ValueError:
                level_name = record.levelno
            frame, depth = logging.currentframe(), 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(level_name, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=level)


def get_logger(name: Optional[str] = None):
    """Return a contextual Loguru logger."""
    return logger.bind(name=name) if name else logger
