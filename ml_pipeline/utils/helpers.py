import json
from datetime import datetime
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    """Return a UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def save_json(data: Any, output_path: Path) -> Path:
    """Save a Python object as pretty JSON and return the saved path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    return output_path
