from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Any, FrozenSet, Iterator
from pathlib import Path
from .response import Response 

Module = Any

class Method(Enum):
    CHECK = auto()
    FIX = auto()

@dataclass(frozen=True)
class Specification:
    name: str
    func: Callable[[Any], Response]
    supported_methods: FrozenSet[Method]
    requires: FrozenSet[str]

    @classmethod
    def create(cls, name: str, func: Callable[[Any], Response], supported_methods: Iterator[Method], requires: Iterator[str] = frozenset()):
        return cls(
            name=name,
            func=func,
            supported_methods=frozenset(supported_methods),
            requires=frozenset(requires)
        )
