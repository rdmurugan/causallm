
import json
import yaml
import os
import logging
from typing import Any

def load_json(file_path: str) -> Any:
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: Any, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    logging.info(f"Saving to {file_path}")
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_yaml(file_path: str) -> Any:
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data: Any, file_path: str) -> None:
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
