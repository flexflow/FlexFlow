from enum import Enum, auto
from typing import FrozenSet, TypeVar, Set, Optional, Iterator, Generic, Iterable, Tuple, Dict
from dataclasses import dataclass, field
from tooling.layout.path import AbsolutePath
import logging

_l = logging.getLogger(__name__)

T = TypeVar('T')

def union(it: Iterator[FrozenSet[T]]) -> Set[T]:
    result: Set[T] = set()
    result.union(set(v) for v in it)
    return result

@dataclass(frozen=True)
class Rule(Generic[T]):
    pre_condition: FrozenSet[T]
    post_condition: T

@dataclass
class Solver(Generic[T]):
    _rules: Set[Rule[T]] = field(default_factory=set)

    def add_conjunction_rule(self, rhs: T, lhs: Iterable[T]) -> None:
        self._rules.add(Rule(pre_condition=frozenset(lhs), post_condition=rhs))

    def add_conjunction_rules(self, rules: Iterable[Tuple[T, Iterable[T]]]) -> None:
        for post, pre in rules:
            self.add_conjunction_rule(post, pre)

    def add_disjunction_rule(self, rhs: T, lhs: Iterable[T]) -> None:
        for ll in lhs:
            self.add_conjunction_rule(rhs, [ll])

    def add_disjunction_rules(self, rules: Iterable[Tuple[T, Iterable[T]]]) -> None:
        for post, pre in rules:
            self.add_disjunction_rule(post, pre)

    def add_alias(self, lhs: Iterable[T], rhs: Iterable[T]) -> None:
        lhs = set(lhs)
        rhs = set(rhs)
        for ll in lhs:
            self.add_conjunction_rule(ll, rhs)
        for rr in rhs:
            self.add_conjunction_rule(rr, lhs)

    def add_aliases(self, aliases: Iterable[Tuple[Iterable[T], Iterable[T]]]) -> None:
        for aa in aliases:
            self.add_alias(*aa)
        
    def add_fact(self, fact: T) -> None:
        self.add_conjunction_rule(fact, [])

    def add_facts(self, facts: Iterable[T]) -> None:
        for fact in facts:
            self.add_fact(fact)

    def solve(self) -> FrozenSet[T]:
        solution: Set[T] = set()
        old_solution: Optional[Set[T]] = None

        rules = set(self._rules)

        while solution != old_solution:
            old_solution = set(solution)
            for rule in list(rules):
                if rule.post_condition in old_solution:
                    rules.remove(rule)
                elif all(pre in old_solution for pre in rule.pre_condition):
                    solution.add(rule.post_condition)

        return frozenset(solution)

_SOLVER = FileAttribute._solver()

_file_attributes: Dict[AbsolutePath, 'FileAttributes'] = {}

@dataclass
class FileAttributes:
    path: AbsolutePath
    _attrs: Set[FileAttribute] = field(default_factory=set)
    _needs_update: bool = False

    def add_fact(self, fact: FileAttribute) -> None:
        _needs_update = True

    @property
    def attrs(self) -> FrozenSet[FileAttribute]:
        if self._needs_update:
            self._update()
        return frozenset(self._attrs)

    def implies(self, fact: FileAttribute) -> bool:
        if self._needs_update:
            self._update()
        return fact in self._attrs

    def implies_all_of(self, facts: Iterable[FileAttribute]) -> bool:
        return all(self.implies(fact) for fact in facts)

    def implies_any_of(self, facts: Iterable[FileAttribute]) -> bool:
        return any(self.implies(fact) for fact in facts)

    def _update(self) -> None:
        solver = FileAttribute._solver()
        solver.add_facts(self._attrs)
        self._attrs = set(solver.solve())
        self._needs_update = False

    @property
    def extension(self) -> Optional[str]:
        if self.implies(FileAttribute.HEADER):
            return '.h'
        elif self.implies(FileAttribute.IS_CUDA_KERNEL):
            return '.cu'
        elif self.implies(FileAttribute.IS_HIP_KERNEL):
            return '.cpp'
        elif self.implies(FileAttribute.CPP_TEST):
            return '.test.cc'
        elif self.implies(FileAttribute.CPP_SOURCE):
            return '.cc'
        elif self.implies(FileAttribute.PYTHON):
            return '.py'
        elif self.implies(FileAttribute.C):
            return '.c'
        else:
            _l.debug('Unknown extension for file attribute set %s', self.attrs)
            return None

    @classmethod
    def for_path(cls, p: AbsolutePath) -> 'FileAttributes':
        assert p.is_file()
        if p not in _file_attributes:
            _file_attributes[p] = FileAttributes(path=p, )
        return _file_attributes[p]
