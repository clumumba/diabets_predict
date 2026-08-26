import os
import sys
import logging
from pathlib import Path

logging_str="[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

project_root = Path(__file__).resolve().parents[2]
log_dir = project_root / "logs"
log_filepath = log_dir / "logging.log"
os.makedirs(log_dir,exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger=logging.getLogger("diabetesPredict")
