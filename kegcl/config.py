from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str, dataset: Optional[str] = None) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if dataset is not None:
        config = _merge(config, config.get("datasets", {}).get(dataset, {}))
        config["dataset"] = dataset
    project_root = config_path.parent.parent
    config["project_root"] = str(project_root)
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = project_root / output_root
    config["output_root"] = str(output_root.resolve())
    data_root = Path(config["data_root"])
    if not data_root.is_absolute():
        data_root = project_root / data_root
    config["data_root"] = str(data_root.resolve())
    return config
