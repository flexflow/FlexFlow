from tooling.layout.file_type_inference.rules.rule import Rule
from tooling.layout.path import Path
from dataclasses import dataclass
from typing import FrozenSet, Type, Dict, Generic, TypeVar, Callable

T = TypeVar('T')

class Runner(Generic[T]):
    get_inputs: Callable[[T], FrozenSet[T]]
    get_outputs: Callable[[T], FrozenSet[T]]

    def create_dependency_graph(self, things: FrozenSet[T]) -> Dict[T, FrozenSet[T]]:
        ...

    def is_acyclic(self): 
        pass


@dataclass(frozen=True)
class RuleCollection:
    rules: FrozenSet[Type[Rule]]

    def run(self, root: AbsolutePath):
        dependency_graph: Dict[Type[Rule], FrozenSet[Type[Rule]]] = {}
