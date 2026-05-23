import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "logs" / "app.log"

LOG_FILE.parent.mkdir(exist_ok=True)

def setup_logger(name = "thyrovision"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)   

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()
