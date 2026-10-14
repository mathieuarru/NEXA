"""Logger factory configured from LOG_LEVEL."""
import logging

from nexa.config import params

def get_logger(name: str) -> logging.Logger:
    """Return a logger for ``name`` at the configured level, adding a stream handler on first use."""
    logger = logging.getLogger(name)

    log_level = params.LOG_LEVEL
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
