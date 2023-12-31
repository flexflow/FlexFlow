from dataclasses import dataclass
from typing import Tuple, Iterable, Union

@dataclass(frozen=True)
class StringLiteral:
    value: str

    def __repr__(self) -> str:
        return f's{repr(self.value)}'

@dataclass(frozen=True)
class NameLiteral:
    name: str

    def __repr__(self) -> str:
        return f'n`{self.name}`'

@dataclass(frozen=True)
class SExpr:
    children: Tuple['AST', ...]

    @staticmethod
    def from_iter(it: Iterable['AST']) -> 'SExpr':
        return SExpr(tuple(it))

    def __repr__(self) -> str:
        return 'x( ' + ' '.join(map(repr, self.children)) + ' )x'


AST = Union[SExpr, NameLiteral, StringLiteral]
