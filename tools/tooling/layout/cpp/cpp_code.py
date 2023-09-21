from dataclasses import dataclass
from ..path import AbsolutePath
from .file_group.file_group_path import FileGroupPath
from ..file_type import FileType
from .library import Library, is_library_root 
from typing import Optional, Iterator, FrozenSet, Container
import functools

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..project import Project

@dataclass(frozen=True)
class CppCode:
    root_path: AbsolutePath
    project: 'Project'

    @property
    def blacklisted(self) -> FrozenSet[AbsolutePath]:
        return frozenset({
            self.lib_dir / 'compiler',
            self.lib_dir / 'ffi',
            self.lib_dir / 'kernels',
            self.lib_dir / 'op-attrs',
            self.lib_dir / 'pcg',
            self.lib_dir / 'runtime',
            self.lib_dir / 'substitutions'
        })

    def get_project_relative_logical_path(self, file_path: AbsolutePath) -> FileGroupPath['Project']:
        return FileGroupPath.create(self.root_path, file_path)

    @property
    def lib_dir(self) -> AbsolutePath:
        return self.project.lib_dir

    @property
    def clang_format_config_path(self) -> AbsolutePath:
        return self.root_path / '.clang-format-for-format-sh'

    @functools.lru_cache()
    def find_libraries(self) -> FrozenSet[Library]:
        def iter_find_libraries(
                within: AbsolutePath, 
                to_skip: Container[AbsolutePath]
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

    # def find_files_of_type(self, file_types: frozenset[FileType]) -> frozenset[AbsolutePath]:
    #     return find_in_path_with_file_type(
    #         path=self.lib_dir,
    #         file_types=file_types,
    #         skip_directories=self.blacklisted
    #     )

    def all_files(self, file_types: Iterator[FileType]) -> FrozenSet[AbsolutePath]:
        _file_types = frozenset(file_types)
        assert all(ft.implies(FileType.CPP) for ft in _file_types)

        raise NotImplementedError()

