import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

from diabetesPredictor import logger
from diabetesPredictor.config.configuration import PROJECT_ROOT, model_path, trained_data


def load_params(params_path: Path) -> dict[str, int]:
    try:
        with open(params_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)["model"]
        return {"n_estimators": int(config["n_estimators"]), "random_state": int(config["random_state"])}
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise RuntimeError(f"Invalid model parameters in {params_path}") from error


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(data_path)
    except (OSError, pd.errors.ParserError) as error:
        raise RuntimeError(f"Error loading data from {data_path}") from error


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "Outcome" not in data.columns:
        raise ValueError("Training data must contain an 'Outcome' column")
    return data.drop(columns="Outcome"), data["Outcome"]


def train_model(features: pd.DataFrame, target: pd.Series, n_estimators: int, random_state: int) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(features, target)
    return model


def save_model(model: RandomForestClassifier, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        pickle.dump(model, file)


def main() -> None:
    params = load_params(PROJECT_ROOT / "params.yml")
    features, target = prepare_data(load_data(trained_data))
    model = train_model(features, target, **params)
    save_model(model, model_path)
    logger.info("Model trained and saved successfully to %s", model_path)


if __name__ == "__main__":
    main()
