from dataclasses import dataclass, field
from typing import (
    Set, 
    Dict, 
    TypeVar, 
    Generic, 
    FrozenSet,
)
import logging
from functools import lru_cache

_l = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class DiGraph(Generic[T]):
    """ 
    A basic self-contained directed graph class. 

    Added here to avoid an additional dependency on networkx, etc.
    """
    _nodes: Set[T] = field(default_factory=set)
    _connectivity: Dict[T, Set[T]] = field(default_factory=dict)
    _reverse_connectivity: Dict[T, Set[T]] = field(default_factory=dict)

    def add_node(self, node: T) -> None:
        self._nodes.add(node)
        if node not in self._connectivity:
            self._connectivity[node] = set()
            self._reverse_connectivity[node] = set()

    def add_edge(self, src: T, dst: T) -> None:
        self.add_node(src)
        self.add_node(dst)
        self._connectivity[src].add(dst)
        self._reverse_connectivity[dst].add(src)

    def _inplace_transitive_closure(self) -> None:
        did_update = True
        while did_update:
            did_update = False
            for src, dsts in self._connectivity.items():
                for dst in dsts.copy():
                    if len(self._connectivity[dst] - self._connectivity[src]) > 0:
                        self._connectivity[src].update(self._connectivity[dst])
                        did_update = True

    def transitive_closure(self) -> "DiGraph[T]":
        result = self.copy()
        result._inplace_transitive_closure()
        return result

    def __repr__(self) -> str:
        return "\n\n".join(["DiGraph {", *(f"  {k} --> {v}" for k, v in self._connectivity.items()), "}"])

    @property
    def nodes(self) -> FrozenSet[T]:
        return frozenset(self._nodes)

    @property
    def edges(self) -> FrozenSet[Tuple[T, T]]:
        return frozenset(
            sum(  # type: ignore
                [[(src, dst) for dst in dsts] for src, dsts in self._connectivity.items()], start=list()
            )
        )

    def dot(self) -> str:
        lines = []
        lines.append("digraph {")
        node_nums = {}
        for i, node in enumerate(self.nodes):
            node_nums[node] = i
            lines.append(f'  n{i} [label="{node}"];')
        for src, dst in self.edges:
            lines.append(f"  n{node_nums[src]} -> n{node_nums[dst]};")
        lines.append("}")
        return "\n".join(lines)

    def is_acyclic(self) -> bool:
        _l.debug(f"Checking presence of cycles in \n{self}")
        tr = self.transitive_closure()
        for src, dsts in tr._connectivity.items():
            if src in dsts:
                return False
        return True

    def copy(self) -> "DiGraph[T]":
        return DiGraph(_nodes=set(self.nodes), _connectivity={k: set(v) for k, v in self._connectivity.items()})

    def topological_order(self) -> Iterator[T]:
        assert self.is_acyclic()

        queue = list(self.nodes)
        visited: Set[T] = set()
        while len(queue) > 0:
            node = queue.pop(0)
            if node in visited:
                continue
            if not visited.issuperset(self._reverse_connectivity[node]):
                continue
            yield node
            visited.add(node)
            for dst in self._connectivity[node]:
                if dst not in visited:
                    queue.append(dst)

    def predecessors(self, node: T) -> FrozenSet[T]:
        return frozenset(self._reverse_connectivity[node])

    @lru_cache(maxsize=None)
    def _ancestors(self, node: T) -> FrozenSet[T]:
        result = {node}
        for p in self.predecessors(node):
            result.update(self._ancestors(p))
        return frozenset(result)

    def ancestors(self, node: T) -> FrozenSet[T]:
        result = self._ancestors(node)
        self._ancestors.cache_clear()
        return result

