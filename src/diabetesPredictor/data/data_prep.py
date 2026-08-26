from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from diabetesPredictor import logger
from diabetesPredictor.config.configuration import PROJECT_ROOT, processed_data, test_data, trained_data


def load_data(filepath: Path) -> pd.DataFrame:
    logger.info("Reading the CSV file %s", filepath)
    return pd.read_csv(filepath)


def load_params(filepath: Path) -> dict[str, float | int]:
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)["data_prep"]
        return {"test_size": float(config["test_size"]), "random_state": int(config["random_state"])}
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise RuntimeError(f"Invalid data-preparation parameters in {filepath}") from error


def split(data: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Splitting processed data into train and test sets")
    return train_test_split(data, test_size=test_size, random_state=random_state)


def main() -> None:
    params = load_params(PROJECT_ROOT / "params.yml")
    train_frame, test_frame = split(load_data(processed_data), **params)
    trained_data.parent.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(trained_data, index=False)
    test_frame.to_csv(test_data, index=False)
    logger.info("Saved train data to %s", trained_data)
    logger.info("Saved test data to %s", test_data)


if __name__ == "__main__":
    main()
