from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml

from sign_motion_refinement.paths import PROJECT_ROOT


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_string(value):
    variables = {"PROJECT_ROOT": str(PROJECT_ROOT), **os.environ}

    def replace(match):
        name, default = match.groups()
        if name in variables:
            return variables[name]
        if default is not None:
            return default
        raise KeyError(f"Configuration references unset environment variable {name!r}")

    return _ENV_PATTERN.sub(replace, value)


def expand_config_values(value):
    """Expand ``${NAME}`` and ``${NAME:-default}`` recursively."""

    if isinstance(value, dict):
        return {key: expand_config_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_config_values(item) for item in value]
    if isinstance(value, str):
        return _expand_string(value)
    return value


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(handle)
        else:
            payload = json.load(handle)
    return expand_config_values(payload)


def deep_update(base, updates):
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out
