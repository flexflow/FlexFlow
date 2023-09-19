from pathlib import Path
from sqlitedict import SqliteDict
import os
import json
from ..data_model.json import Json
from typing import Mapping


def get_cache_dir() -> Path:
    path = Path.home() / os.environ.get("XDG_CACHE_HOME", ".cache") / "ff" / "issue-triage"
    path.mkdir(exist_ok=True, parents=True)
    return path


def get_cache() -> Mapping[str, Json]:
    return SqliteDict(get_cache_dir() / "cache.sqlite", encode=json.dumps, decode=json.loads)
