from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s: %(message)s]')

project_name = "diabetesPredictor"

list_files = [
    ".github/workflows/.gitkeep",
    f"Data/__init__.py",
    f"Data/raw/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/data/__init__.py",
    f"src/{project_name}/data/data_collection.py",
    f"src/{project_name}/data/data_prep.py",
    f"src/{project_name}/model/__init__.py",
    f"src/{project_name}/model/model_build.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    "notebook/research.ipynb",
    "requirements.txt",
    "params.yml",
    "config.py"
]

for filepath in list_files:
    filepath = Path(filepath)

    if filepath.parent != Path("."):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Created folder %s", filepath.parent)

    if not filepath.exists():
        filepath.touch()
        logging.info("Created empty file %s", filepath)
    else:
        logging.info("File %s already exists", filepath)

    
