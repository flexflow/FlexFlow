from pathlib import Path 
from dataclasses import dataclass
from .path import AbsolutePath, LogicalPath
from .file_type import find_in_path_with_file_type , FileType
from .library import Library, is_library_root 
from typing import Optional, Iterator
import subprocess
import functools
from enum import Enum, auto

class Language(Enum):
    CXX = auto()
    CUDA = auto()
    HIP = auto()
    PYTHON = auto()
    C = auto()

@dataclass(frozen=True)
class Project:
    root_path: AbsolutePath

    @property
    def blacklisted(self) -> frozenset[AbsolutePath]:
        return frozenset({
            self.lib_dir / 'compiler',
            self.lib_dir / 'ffi',
            self.lib_dir / 'kernels',
            self.lib_dir / 'op-attrs',
            self.lib_dir / 'pcg',
            self.lib_dir / 'runtime',
            self.lib_dir / 'substitutions'
        })

    def get_project_relative_logical_path(self, file_path: AbsolutePath) -> LogicalPath['Project']:
        return LogicalPath.create(self.root_path, file_path)

    @property
    def lib_dir(self) -> AbsolutePath:
        return self.root_path / 'lib'

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
    def clang_format_config_path(self) -> AbsolutePath:
        return self.root_path / '.clang-format-for-format-sh'

    @functools.cache
    def find_libraries(self) -> frozenset[Library]:
        def iter_find_libraries(
                within: AbsolutePath, 
                to_skip: frozenset[AbsolutePath]
        ) -> Iterator[Library]:
            if within.is_file():
                return

            if within in to_skip:
                return
            if is_library_root(within):
                yield Library.create(root_path=within, project=self)
            else:
                for p in within.iterdir():
                    yield from iter_find_libraries(p, to_skip)

        return frozenset(iter_find_libraries(self.lib_dir, self.blacklisted))


    def get_containing_library(self, path: AbsolutePath) -> Optional[Library]:
        for library in self.find_libraries():
            if library.contains(path):
                return library
        return None

    @classmethod
    def for_path(cls, p: Path) -> 'Project':
        abs_path = AbsolutePath(subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel']
        ).decode().strip())
        assert abs_path.is_dir()
        return cls(root_path=abs_path)

    def get_unstaged_changes(self, *, exclude_submodules=True) -> frozenset[AbsolutePath]:
        output = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=self.root_path).decode()
        result = [AbsolutePath(line[3:]) for line in output.splitlines()]
        if exclude_submodules:
            result = [path for path in result if not path.is_relative_to(self.deps_dir)]
        return frozenset(result)

    def find_files_of_type(self, file_types: frozenset[FileType]) -> frozenset[AbsolutePath]:
        return find_in_path_with_file_type(
            path=self.lib_dir,
            file_types=file_types,
            skip_directories=self.blacklisted
        )

    def all_files(self, languages: frozenset[Language]) -> frozenset[AbsolutePath]:
        all_handled_languages = frozenset({Language.CXX})
        unhandled_languages = languages - all_handled_languages 
        if len(unhandled_languages) > 0:
            raise ValueError(f'Unhandled languages: {unhandled_languages}')

        def _iter(lang: Language):
            if Language.CXX in languages:
                yield from self.find_files_of_type(FileType.all())

        result = frozenset()
        for lang in languages:
            result |= frozenset(_iter(lang))
        return result
