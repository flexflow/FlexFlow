from dataclasses import dataclass
from .cpp.cpp_code import CppCode
from .path import AbsolutePath
from typing import FrozenSet, Iterator
import subprocess
from pathlib import Path
from .file_type import FileType

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

    @classmethod
    def for_path(cls, p: Path) -> 'Project':
        abs_path = AbsolutePath(subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=p
        ).decode().strip())
        assert abs_path.is_dir()
        return cls(root_path=abs_path)

    def all_files(self, file_type: Iterator[FileType]) -> FrozenSet[AbsolutePath]:
        _file_type = frozenset(file_type)
        result: FrozenSet[AbsolutePath] = frozenset()
        cpp_files = {ft for ft in _file_type if ft.implies(FileType.CPP)}
        result |= self.cpp_code.all_files(cpp_files)

        handled = cpp_files
        unhandled = _file_type - handled

        if len(unhandled) > 0:
            raise ValueError(f'Unhandled file_type: {unhandled}')

        return result
