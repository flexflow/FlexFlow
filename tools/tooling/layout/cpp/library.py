from dataclasses import dataclass
from ..path import AbsolutePath
from .file_group.file_group import FileGroup
from .file_group.file_group_path import FileGroupPath
from typing import Set, FrozenSet

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .cpp_code import CppCode 

def include_path(library_root: AbsolutePath) -> AbsolutePath:
    return library_root / 'include'

def src_path(library_root: AbsolutePath) -> AbsolutePath:
    return library_root / 'src'

def cmake_file_path(library_root: AbsolutePath) -> AbsolutePath:
    return library_root / 'CMakeLists.txt'

def is_library_root(path: AbsolutePath) -> bool:
    return all([
        include_path(path).is_dir(),
        src_path(path).is_dir(),
        cmake_file_path(path).is_file(),
    ])

def get_library_name(library_root: AbsolutePath) -> str:
    return ''

@dataclass(frozen=True)
class Library:
    name: str
    root_path: AbsolutePath
    cpp_code: 'CppCode'

    @property
    def src_path(self) -> AbsolutePath:
        return src_path(self.root_path)

    @property
    def include_path(self) -> AbsolutePath:
        return include_path(self.root_path)

    @property
    def cmake_file_path(self) -> AbsolutePath:
        return include_path(self.root_path)

    def get_logical_path(self, file_path: AbsolutePath) -> FileGroup:
        assert self.is_valid_path(file_path), (file_path, self.root_path)
        if file_path.is_relative_to(self.src_path):
            base = self.src_path
        else:
            base = self.include_path
        return FileGroup.create(base, file_path)

    def contains(self, file_path: AbsolutePath) -> bool:
        return file_path.is_relative_to(self.root_path)

    def is_valid_path(self, file_path: AbsolutePath) -> bool:
        if file_path.is_relative_to(self.include_path) and file_type_of_path(file_path) == FileType.HEADER:
            return True
        if file_path.is_relative_to(self.src_path) and file_type_of_path(file_path) is not None:
            return True
        return False

    def get_library_relative_logical_path(self, file_path: AbsolutePath) -> FileGroupPath['Library']:
        return FileGroupPath.create(self.root_path, file_path)

    def find_logical_files(self) -> FrozenSet[LogicalFile]:
        logical_files: Set[LogicalFile] = set()
        for p in find_in_path_with_file_type(self.root_path, file_types=FileType.all()):
            if self.is_valid_path(p):
                logical_files.add(LogicalFile.create(p, self))
        return frozenset(logical_files)

    @classmethod
    def create(cls, root_path: AbsolutePath, cpp_code: 'CppCode') -> 'Library':
        assert is_library_root(root_path)
        return cls(
            name=get_library_name(root_path),
            root_path=root_path,
            cpp_code=cpp_code
        )

