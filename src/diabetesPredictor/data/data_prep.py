from pathlib import Path
from typing import TypedDict

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from diabetesPredictor import logger
from diabetesPredictor.config.configuration import (
    PROJECT_ROOT,
    processed_data,
    x_test_data,
    x_train_data,
    y_test_data,
    y_train_data,
)


class DataPrepParams(TypedDict):
    test_size: float
    random_state: int


def load_data(filepath: Path) -> pd.DataFrame:
    logger.info("Reading the CSV file %s", filepath)
    return pd.read_csv(filepath)


def load_params(filepath: Path) -> DataPrepParams:
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)["data_prep"]
        return {"test_size": float(config["test_size"]), "random_state": int(config["random_state"])}
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise RuntimeError(f"Invalid data-preparation parameters in {filepath}") from error


def split(
    data: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if "Outcome" not in data.columns:
        raise ValueError("Processed data must contain an 'Outcome' column")

    logger.info("Splitting processed data into train and test sets")
    features = data.drop(columns="Outcome")
    target = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def main() -> None:
    params = load_params(PROJECT_ROOT / "params.yml")
    X_train, X_test, y_train, y_test = split(load_data(processed_data), **params)
    x_train_data.parent.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(x_train_data, index=False)
    X_test.to_csv(x_test_data, index=False)
    y_train.to_frame(name="Outcome").to_csv(y_train_data, index=False)
    y_test.to_frame(name="Outcome").to_csv(y_test_data, index=False)
    logger.info("Saved X_train to %s", x_train_data)
    logger.info("Saved X_test to %s", x_test_data)
    logger.info("Saved y_train to %s", y_train_data)
    logger.info("Saved y_test to %s", y_test_data)


if __name__ == "__main__":
    main()
