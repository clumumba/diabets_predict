from pathlib import Path

import pandas as pd

from diabetesPredictor import logger
from diabetesPredictor.config.configuration import file_path, processed_data


def load_data(input_path: Path) -> pd.DataFrame:
    try:
        data = pd.read_csv(input_path)
        logger.info("Raw data loaded from %s", input_path)
        return data
    except (OSError, pd.errors.ParserError) as error:
        raise RuntimeError(f"Failed to load raw data from {input_path}") from error


def save_data(data: pd.DataFrame, output_path: Path) -> Path:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        logger.info("Saved collected data to %s", output_path)
        return output_path
    except OSError as error:
        raise RuntimeError(f"Could not save data to {output_path}") from error


if __name__ == "__main__":
    save_data(load_data(file_path), processed_data)
