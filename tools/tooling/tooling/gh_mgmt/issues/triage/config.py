from dataclasses import dataclass, field
from datetime import timedelta, datetime, date
from pathlib import Path
import subprocess
from typing import Optional, ClassVar
import json
from .data_model.json import Json
from typing import Mapping
from sqlitedict import SqliteDict
import os

def get_auth_token_from_gh_cli() -> Optional[str]:
    try:
        return subprocess.check_output(["gh", "auth", "token"]).decode().strip()
    except subprocess.CalledProcessError:
        return None

def get_default_cache_dir() -> Path:
    return Path.home() / os.environ.get("XDG_CACHE_HOME", ".cache") / "ff" / "issue-triage"

def to_datetime(d: date):
    return datetime.combine(d, datetime.min.time())

def get_cache_path() -> Path:
    assert TriageConfig._CACHE_DIR is not None
    TriageConfig._CACHE_DIR.mkdir(exist_ok=True, parents=True)
    return TriageConfig._CACHE_DIR / "cache.sqlite"

def get_cache() -> Mapping[str, Json]:
    return SqliteDict(get_cache_path(), encode=json.dumps, decode=json.loads)

def get_now() -> datetime:
    assert TriageConfig._NOW is not None
    return TriageConfig._NOW

def get_max_lookback() -> timedelta:
    assert TriageConfig._MAX_LOOKBACK is not None
    return TriageConfig._MAX_LOOKBACK

def get_beginning_of_time() -> datetime:
    return get_now() - get_max_lookback()

def get_auth_token() -> str:
    assert TriageConfig._AUTH_TOKEN is not None
    return TriageConfig._AUTH_TOKEN

class TriageConfig:
    _MAX_LOOKBACK: Optional[timedelta] = None
    _CACHE_DIR: Optional[Path] = None
    _AUTH_TOKEN: Optional[str] = None
    _NOW: datetime = datetime.now()

def init_config(max_lookback: timedelta=None, cache_dir: Path=None, auth_token: str=None):
    TriageConfig._MAX_LOOKBACK = max_lookback
    TriageConfig._CACHE_DIR = cache_dir 
    TriageConfig._AUTH_TOKEN = auth_token 
