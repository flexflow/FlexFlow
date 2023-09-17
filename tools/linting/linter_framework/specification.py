from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Any
from pathlib import Path

Module = Any

class Method(Enum):
    CHECK = auto()
    FIX = auto()

@dataclass(frozen=True)
class Specification:
    name: str
    make_args: Callable[[Module, LinterMethod], Any]
    source_path: Path
    supported_methods: frozenset[LinterMethod]
    requires: frozenset[str] = frozenset()
