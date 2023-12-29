from dataclasses import dataclass
from typing import Callable, FrozenSet, Iterable
from tooling.linting.framework.response import Response
from tooling.layout.project import Project
from tooling.linting.framework.method import Method
from tooling.linting.framework.settings import Settings


@dataclass(frozen=True)
class Specification:
    name: str
    func: Callable[[Settings, Project, Method], Response]
    supported_methods: FrozenSet[Method]
    requires: FrozenSet[str]

    @classmethod
    def create(
        cls,
        name: str,
        func: Callable[[Settings, Project, Method], Response],
        supported_methods: Iterable[Method],
        requires: Iterable[str] = frozenset(),
    ) -> "Specification":
        return cls(name=name, func=func, supported_methods=frozenset(supported_methods), requires=frozenset(requires))
