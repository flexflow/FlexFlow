from abc import ABC, abstractmethod
from tooling.layout.file_type_inference.rules.rule import Rule, ExprExtra
from tooling.layout.path import AbsolutePath, children
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from dataclasses import dataclass, field
from typing import Dict, Generic, TypeVar, Set, DefaultDict, Union, Iterator, Optional, FrozenSet, Iterable, Any, Callable, List, Sequence, Tuple
from collections import defaultdict
import logging

_l = logging.getLogger(__name__)

T = TypeVar('T')

class ExprExtraBackend(ABC):
    @abstractmethod
    def for_rule(self, rule: Rule) -> ExprExtra:
        ...

    @abstractmethod
    def result(self) -> Callable[[FileAttribute], Sequence[Any]]:
        ...

@dataclass
class DictBackend(ExprExtraBackend):
    _d: DefaultDict[FileAttribute, List[Any]] = field(default_factory=lambda: defaultdict(list))

    def for_rule(self, rule: Rule) -> ExprExtra:
        def save_func(to_save: Any, backend: 'DictBackend' = self, rule: Rule = rule) -> None:
            self._d[rule.result].append(to_save)
        return ExprExtra(save_func)

    def result(self) -> Callable[[FileAttribute], Sequence[Any]]:
        return lambda attr: self._d[attr]


@dataclass
class DiGraph(Generic[T]):
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

    def transitive_closure(self) -> 'DiGraph[T]':
        result = self.copy()
        result._inplace_transitive_closure()
        return result

    def __repr__(self) -> str:
        return '\n\n'.join([
            'DiGraph {',
            *(f'  {k} --> {v}' for k, v in self._connectivity.items()),
            '}'
        ])

    @property
    def nodes(self) -> FrozenSet[T]:
        return frozenset(self._nodes)

    @property
    def edges(self) -> FrozenSet[Tuple[T, T]]:
        return frozenset(sum(
            [[(src, dst) for dst in dsts] for src, dsts in self._connectivity.items()],
            start=list()
        ))

    def dot(self) -> str:
        lines = []
        lines.append('digraph {')
        node_nums = {}
        for i, node in enumerate(self.nodes):
            node_nums[node] = i
            lines.append(f'  n{i} [label="{node}"];')
        for src, dst in self.edges:
            lines.append(f'  n{node_nums[src]} -> n{node_nums[dst]};')
        lines.append('}')
        return '\n'.join(lines)


    def is_acyclic(self) -> bool:
        _l.debug(f'Checking presence of cycles in \n{self}')
        tr = self.transitive_closure()
        for src, dsts in tr._connectivity.items():
            if src in dsts:
                return False
        return True

    def copy(self) -> 'DiGraph[T]':
        return DiGraph(_nodes=set(self.nodes), _connectivity={k : set(v) for k, v in self._connectivity.items()})

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

@dataclass(frozen=True)
class RuleCollection:
    rules: FrozenSet[Rule]

    def run(self, root: AbsolutePath) -> 'InferenceResult':
        backend = DictBackend()
        dependency_graph: DiGraph[Union[Rule, FileAttribute]] = DiGraph()

        for rule in self.rules:
            dependency_graph.add_node(rule)
            for inp in rule.inputs:
                _l.debug(f'Adding dependency from {inp} to {rule.name}')
                dependency_graph.add_edge(inp, rule)
            for out in rule.outputs:
                _l.debug(f'Adding dependency from {rule.name} to {out}')
                dependency_graph.add_edge(rule, out)
        _l.debug('Built dependency graph:\n' + dependency_graph.dot())

        _l.debug('Checking dependency graph for cycles')
        assert dependency_graph.is_acyclic()

        all_children = list(children(root))
        attrs: DefaultDict[AbsolutePath, FrozenSet[FileAttribute]] = defaultdict(frozenset)
        for node in dependency_graph.topological_order():
            if isinstance(node, Rule):
                _l.debug(f'Running rule {node.signature}')
                extra = backend.for_rule(node)
                num_added = 0
                for p in all_children:
                    if node.condition.evaluate(p, lambda path: attrs[path], extra=extra):
                        num_added += 1
                        attrs[p] |= node.outputs
                _l.debug(f'Rule {node.name} found {num_added} paths that satisfy {node.result}')
        return InferenceResult(dict(attrs), get_saved=backend.result())

class InferenceResult:
    def __init__(self, attrs: Dict[AbsolutePath, FrozenSet[FileAttribute]], get_saved: Callable[[FileAttribute], Any]) -> None:
        self._attrs = attrs
        self._reverse_attrs: DefaultDict[FileAttribute, Set[AbsolutePath]] = defaultdict(set)
        self._get_saved = get_saved
        for k, v in attrs.items():
            for a in v:
                self._reverse_attrs[a].add(k)

    def get_saved(self, attr: FileAttribute) -> Any:
        return self._get_saved(attr)

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

