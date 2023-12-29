from tooling.layout.cpp.cpp_code import CppCode
from tooling.layout.path import AbsolutePath
from tooling.layout.file_type_inference.all_rules import all_rules
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from tooling.layout.file_type_inference.rules.rule import Rule
from tooling.layout.file_type_inference.rules.infer import InferenceResult, RuleCollection
from typing import FrozenSet, Optional, Iterator, Callable, Iterable
import subprocess
from pathlib import Path
import string


class Project:
    def __init__(self, root_path: AbsolutePath):
        self.root_path = root_path
        self._file_types: Optional[InferenceResult] = None
        self._rules: FrozenSet[Rule] = all_rules

    @property
    def file_types(self) -> InferenceResult:
        if self._file_types is None:
            predefined = {self.root_path: frozenset({FileAttribute.IS_PROJECT_ROOT})}
            self._file_types = RuleCollection(predefined, self._rules).run(root=self.root_path)
        return self._file_types

    def add_rules(self, rules: Iterable[Rule]) -> None:
        self._rules = self._rules.union(rules)

    @property
    def cpp_code(self) -> CppCode:
        return CppCode(self.root_path, self)

    @property
    def deps_dir(self) -> AbsolutePath:
        return self.root_path / "deps"

    @property
    def tools_download_dir(self) -> AbsolutePath:
        return self.root_path / ".tools"

    @property
    def state_dir(self) -> AbsolutePath:
        return self.root_path / ".state"

    @property
    def lib_dir(self) -> AbsolutePath:
        return self.root_path / "lib"

    def get_unstaged_changes(self, *, exclude_submodules: bool = True) -> FrozenSet[AbsolutePath]:
        output = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=self.root_path).decode()
        result = [AbsolutePath(line[3:]) for line in output.splitlines()]
        if exclude_submodules:
            result = [path for path in result if not path.is_relative_to(self.deps_dir)]
        return frozenset(result)

    def include_guard_for_path(self, path: AbsolutePath) -> str:
        return "FLEXFLOW_" + "".join(
            c.upper() if c in string.ascii_letters else "_" for c in str(path.relative_to(self.root_path))
        )

    def files_satisfying(
        self, f: Callable[[AbsolutePath], bool], base_path: Optional[AbsolutePath] = None
    ) -> FrozenSet[AbsolutePath]:
        if base_path is None:
            base_path = self.root_path

        def _recurse(p: AbsolutePath) -> Iterator[AbsolutePath]:
            if p.is_dir():
                for child in p.iterdir():
                    yield from _recurse(child)
            elif p.is_file() and f(p):
                yield p

        return frozenset(_recurse(base_path))

    def directories_satisfying(
        self, f: Callable[[AbsolutePath], bool], base_path: Optional[AbsolutePath] = None, allow_nesting: bool = True
    ) -> FrozenSet[AbsolutePath]:
        if base_path is None:
            base_path = self.root_path

        def _recurse(p: AbsolutePath) -> Iterator[AbsolutePath]:
            if p.is_dir() and f(p):
                yield p
                if not allow_nesting:
                    for child in p.iterdir():
                        yield from _recurse(child)

        return frozenset(_recurse(base_path))

    @property
    def build_directories(self) -> FrozenSet[AbsolutePath]:
        return self.file_types.with_attr(FileAttribute.BUILD_DIRECTORY)

    @classmethod
    def for_path(cls, p: Path) -> "Project":
        if p.is_file():
            p = p.parent
        abs_path = AbsolutePath.create(
            Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=p).decode().strip())
        )
        assert abs_path.is_dir()
        return cls(root_path=abs_path)
