from tooling.layout.file_type_inference.rules.rule import Rule
from tooling.layout.path import AbsolutePath, children
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from dataclasses import dataclass, field
from typing import Dict, Generic, TypeVar, Set, DefaultDict, Union, Iterator, Optional, FrozenSet, Iterable
from collections import defaultdict
import logging

_l = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class DiGraph(Generic[T]):
    nodes: Set[T] = field(default_factory=set)
    connectivity: Dict[T, Set[T]] = field(default_factory=dict)
    reverse_connectivity: Dict[T, Set[T]] = field(default_factory=dict)

    def add_node(self, node: T) -> None:
        self.nodes.add(node)
        if node not in self.connectivity:
            self.connectivity[node] = set()
            self.reverse_connectivity[node] = set()

    def add_edge(self, src: T, dst: T) -> None:
        self.add_node(src)
        self.add_node(dst)
        self.connectivity[src].add(dst)
        self.connectivity[dst].add(src)

    def _inplace_transitive_reduction(self) -> None:
        did_update = True
        while did_update:
            did_update = False
            for src, dsts in self.connectivity.items():
                for dst in dsts:
                    if len(self.connectivity[dst] - self.connectivity[src]) > 0:
                        self.connectivity[src].update(self.connectivity[dst])
                        did_update = True

    def transitive_reduction(self) -> 'DiGraph[T]':
        result = self.copy()
        result._inplace_transitive_reduction()
        return result

    def is_acyclic(self) -> bool:
        tr = self.transitive_reduction()
        for src, dsts in tr.connectivity.items():
            if src in dsts:
                return False
        return True

    def copy(self) -> 'DiGraph[T]':
        return DiGraph(nodes=set(self.nodes), connectivity={k : set(v) for k, v in self.connectivity.items()})

    def topological_order(self) -> Iterator[T]:
        assert self.is_acyclic()
        queue = []
        for dst, srcs in self.reverse_connectivity.items():
            if len(srcs) == 0 and dst not in queue:
                queue.append(dst)

        visited = set()
        while len(queue) > 0:
            node = queue.pop(0)
            if node in visited:
                continue
            yield node
            visited.add(node)
            for dst in self.connectivity[node]:
                if dst not in visited:
                    queue.append(dst)

@dataclass(frozen=True)
class RuleCollection:
    rules: FrozenSet[Rule]

    def run(self, root: AbsolutePath) -> 'InferenceResult':
        dependency_graph: DiGraph[Union[Rule, FileAttribute]] = DiGraph()

        for rule in self.rules:
            dependency_graph.add_node(rule)
            for inp in rule.inputs:
                dependency_graph.add_edge(inp, rule)
            for out in rule.outputs:
                dependency_graph.add_edge(out, rule)

        assert dependency_graph.is_acyclic()

        all_children = list(children(root))
        attrs: DefaultDict[AbsolutePath, FrozenSet[FileAttribute]] = defaultdict(frozenset)
        for node in dependency_graph.topological_order():
            if isinstance(node, Rule):
                _l.debug(f'Running rule {node}')
                for p in all_children:
                    if node.condition.evaluate(p, lambda path: attrs[path]):
                        attrs[p] |= node.outputs
        return InferenceResult(dict(attrs))

class InferenceResult:
    def __init__(self, attrs: Dict[AbsolutePath, FrozenSet[FileAttribute]]) -> None:
        self._attrs = attrs
        self._reverse_attrs: DefaultDict[FileAttribute, Set[AbsolutePath]] = defaultdict(set)
        for k, v in attrs.items():
            for a in v:
                self._reverse_attrs[a].add(k)

    def for_path(self, p: AbsolutePath) -> FrozenSet[FileAttribute]:
        return frozenset(self._attrs[p])

    def with_attr(self, attr: FileAttribute, within: Optional[AbsolutePath] = None) -> FrozenSet[AbsolutePath]:
        if within is None:
            return frozenset(self._reverse_attrs[attr])
        else:
            return frozenset(p for p in self._reverse_attrs[attr] if p.is_relative_to(within))

    def without_attr(self, attr: FileAttribute, within: Optional[AbsolutePath] = None) -> FrozenSet[AbsolutePath]:
        return self._all_paths(within) - self._reverse_attrs[attr]

    def _all_paths(self, within: Optional[AbsolutePath] = None) -> FrozenSet[AbsolutePath]:
        if within is None:
            return frozenset(self._attrs.keys())
        else:
            return frozenset(p for p in self._attrs.keys() if p.is_relative_to(within))

    def without_all_of_attrs(self, attrs: Iterable[FileAttribute], within: Optional[AbsolutePath] = None) -> FrozenSet[AbsolutePath]:
        return self._all_paths(within) - self.with_all_of_attrs(attrs, within=within)

    def without_any_of_attrss(self, attrs: Iterable[FileAttribute], within: Optional[AbsolutePath] = None) -> FrozenSet[AbsolutePath]:
        return self._all_paths(within) - self.with_any_of_attrs(attrs, within=within)

    def with_all_of_attrs(self, attrs: Iterable[FileAttribute], within: Optional[AbsolutePath] = None) -> FrozenSet[AbsolutePath]:
        result: Optional[FrozenSet[AbsolutePath]] = None
        for at in attrs:
            if result is None:
                result = self.with_attr(at, within=within)
            else:
                result &= self.with_attr(at, within=within)

        if result is None:
            raise ValueError('Cannot call with_all_of_attrs on empty attr set')
        else:
            return result

    def with_any_of_attrs(self, attrs: Iterable[FileAttribute], within: Optional[AbsolutePath] = None) -> FrozenSet[AbsolutePath]:
        result: Optional[FrozenSet[AbsolutePath]] = None
        for at in attrs:
            if result is None:
                result = self.with_attr(at, within=within)
            else:
                result |= self.with_attr(at, within=within)

        if result is None:
            raise ValueError('Cannot call with_any_of_attrs on empty attr set')
        else:
            return result

