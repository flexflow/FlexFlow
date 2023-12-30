from dataclasses import dataclass
from typing import Tuple, Iterable, Union

@dataclass(frozen=True)
class StringLiteral:
    value: str

@dataclass(frozen=True)
class NameLiteral:
    name: str

@dataclass(frozen=True)
class SExpr:
    children: Tuple['AST', ...]

    @staticmethod
    def from_iter(it: Iterable['AST']) -> 'SExpr':
        return SExpr(tuple(it))


AST = Union[SExpr, NameLiteral, StringLiteral]
