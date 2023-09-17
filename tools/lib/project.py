from pathlib import Path
from dataclasses import dataclass
from .path import AbsolutePath
from .file_type import find_in_path_with_file_type , FileType
from .library import Library, is_library_root 
from typing import Set, Optional, Iterator
import subprocess
import functools

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

    @property
    def lib_dir(self) -> AbsolutePath:
        return self.root_path / 'lib'

    @functools.cache
    def find_libraries(self) -> Set[Library]:
        return find_libraries(within=self.lib_dir)

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

    def find_files_of_type(self, file_types: frozenset[FileType]) -> frozenset[AbsolutePath]:
        return find_in_path_with_file_type(
            path=self.lib_dir,
            file_types=file_types,
            skip_directories=self.blacklisted
        )

def find_libraries(within: AbsolutePath, to_skip: Optional[Set[AbsolutePath]] = None) -> Set[Library]:
    if to_skip is None:
        to_skip = set()
    return set(iter_find_libraries(within, to_skip))

def iter_find_libraries(
        within: AbsolutePath, 
        to_skip: Set[AbsolutePath]
) -> Iterator[Library]:
    if within.is_file():
        return

    if within in to_skip:
        return
    if is_library_root(within):
        yield Library.create(root_path=within)
    else:
        for p in within.iterdir():
            yield from iter_find_libraries(p, to_skip)

