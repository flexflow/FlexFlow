from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from tooling.layout.file_type_inference.framework.rule import Rule, ExprExtra
from collections import defaultdict
from typing import Any, Sequence, Callable, List, DefaultDict
from tooling.layout.file_type_inference.file_attribute import FileAttribute

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
        def save_func(to_save: Any, backend: "DictBackend" = self, rule: Rule = rule) -> None:
            self._d[rule.result].append(to_save)

        return ExprExtra(save_func)

    def result(self) -> Callable[[FileAttribute], Sequence[Any]]:
        return lambda attr: self._d[attr]

