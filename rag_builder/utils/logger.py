import logging
import sys
from typing import Optional
from ..config.settings import settings

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """로거 인스턴스 생성"""
    logger_name = name or __name__
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, settings.log_level.upper()))
    return logger