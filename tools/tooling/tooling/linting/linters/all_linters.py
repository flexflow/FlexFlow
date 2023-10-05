import tooling.linting.linters.find_missing_files as find_missing_files
import tooling.linting.linters.fix_include_guards as fix_include_guards 
import tooling.linting.linters.clang_format as clang_format
import tooling.linting.linters.clang_tidy as clang_tidy
from tooling.linting.framework.manager import Manager
from tooling.linting.framework.specification import Specification
from tooling.linting.framework.method import Method
from typing import cast, Any
from enum import Enum, auto

class SpecificLinter(Enum):
    missing_files = auto()
    include_guards = auto()
    clang_format = auto()
    clang_tidy = auto()

    @property
    def _module(self) -> Any:
        if self == SpecificLinter.missing_files:
            return find_missing_files
        elif self == SpecificLinter.include_guards:
            return fix_include_guards
        elif self == SpecificLinter.clang_format:
            return clang_format
        elif self == SpecificLinter.clang_tidy:
            return clang_tidy

    @property
    def spec(self) -> Specification:
        return cast(Specification, self._module.spec)

    def get_manager(self) -> Manager:
        return Manager.from_iter([
            self.spec
        ])

    @property
    def supports_fix(self) -> bool:
        return Method.FIX in self.spec.supported_methods

    @property
    def supports_check(self) -> bool:
        return Method.CHECK in self.spec.supported_methods


def all_linters() -> Manager:
    return Manager(frozenset({
        find_missing_files.spec,
        fix_include_guards.spec,
        clang_format.spec,
        clang_tidy.spec
    }))
