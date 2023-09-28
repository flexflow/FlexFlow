from dataclasses import dataclass
from tooling.layout.cpp.cpp_code import CppCode
from tooling.layout.path import AbsolutePath
from typing import FrozenSet, Optional, Iterator, Callable
import subprocess
from pathlib import Path
import string

@dataclass(frozen=True)
class Project:
    root_path: AbsolutePath

    @property
    def cpp_code(self) -> CppCode:
        return CppCode(self.root_path, self)

    @property
    def deps_dir(self) -> AbsolutePath:
        return self.root_path / 'deps'

    @property
    def tools_download_dir(self) -> AbsolutePath:
        return self.root_path / '.tools'

    @property
    def state_dir(self) -> AbsolutePath:
        return self.root_path / '.state'

    @property
    def lib_dir(self) -> AbsolutePath:
        return self.root_path / 'lib'
    
    def get_unstaged_changes(self, *, exclude_submodules: bool = True) -> FrozenSet[AbsolutePath]:
        output = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=self.root_path).decode()
        result = [AbsolutePath(line[3:]) for line in output.splitlines()]
        if exclude_submodules:
            result = [path for path in result if not path.is_relative_to(self.deps_dir)]
        return frozenset(result)

    def include_guard_for_path(self, path: AbsolutePath) -> str:
        return ''.join(c if c in string.ascii_letters else '/'
                for c in str(path.relative_to(self.root_path)))

    def files_satisfying(self, f: Callable[[AbsolutePath], bool], base_path: Optional[AbsolutePath] = None) -> FrozenSet[AbsolutePath]:
        if base_path is None:
            base_path = self.root_path

        def _recurse(p: AbsolutePath) -> Iterator[AbsolutePath]:
            if p.is_dir():
                for child in p.iterdir():
                    yield from _recurse(child)
            elif p.is_file() and f(p):
                yield p
    
        return frozenset(_recurse(self.root_path))

    @classmethod
    def for_path(cls, p: Path) -> 'Project':
        abs_path = AbsolutePath(subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=p
        ).decode().strip())
        assert abs_path.is_dir()
        return cls(root_path=abs_path)
