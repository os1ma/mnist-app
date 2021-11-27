import os
from datetime import datetime


def log_info(message: str) -> None:
    now = datetime.now()
    print(f"[{now}] {message}")


def get_model_tag() -> str:
    return os.environ['MODEL_TAG']
