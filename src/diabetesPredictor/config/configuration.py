from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
file_path = PROJECT_ROOT / "Data" / "raw" / "data.csv"
collected_data = PROJECT_ROOT / "Data" / "interim" / "collected.csv"
processed_data = PROJECT_ROOT / "Data" / "processed" / "processed.csv"
trained_data = PROJECT_ROOT / "Data" / "processed" / "train.csv"
test_data = PROJECT_ROOT / "Data" / "processed" / "test.csv"
model_path = PROJECT_ROOT / "models" / "model.pkl"
metrics_path = PROJECT_ROOT / "metrics" / "metrics.json"
