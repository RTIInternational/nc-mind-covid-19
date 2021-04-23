import json
from pathlib import Path
from typing import Any, Dict


def load_parameters(params_path: Path) -> Dict[str, Any]:
    """Load parameters from the parameters.json path"""
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
    else:
        RuntimeError("Parameters File does not exist.")

    return params
