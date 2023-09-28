from dataclasses import dataclass
from typing import FrozenSet, Callable, Iterator, cast
from tooling.layout.file_type import FileAttribute, FileAttributes
from tooling.layout.cpp.file_group.file_group import FileGroup
from tooling.layout.path import AbsolutePath

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

    def contains(self, file_path: AbsolutePath) -> bool:
        return file_path.is_relative_to(self.root_path)

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

    def is_valid_path(self, file_path: AbsolutePath) -> bool:
        attrs = FileAttributes.for_path(file_path)
        if file_path.is_relative_to(self.include_path) and attrs.implies(FileAttribute.CPP_PUBLIC_HEADER):
            return True
        if file_path.is_relative_to(self.src_path) and attrs.implies_any_of([FileAttribute.CPP, FileAttribute.IMPL]):
            return True
        return False

    def find_file_groups(self) -> FrozenSet[FileGroup]:
        results = set(FileGroup.try_create(path, library=self) for path in self.files_satisfying(
            lambda p: FileAttributes.for_path(p).implies_any_of([
                FileAttribute.CPP_PUBLIC_HEADER,
                FileAttribute.CPP_PRIVATE_HEADER,
                FileAttribute.CPP_SOURCE,
                FileAttribute.CPP_TEST
            ])
        ))
        results.remove(None)
        return frozenset(cast(FrozenSet[FileGroup], results))

    @classmethod
    def create(cls, root_path: AbsolutePath, cpp_code: 'CppCode') -> 'Library':
        assert is_library_root(root_path)
        return cls(
            name=get_library_name(root_path),
            root_path=root_path,
            cpp_code=cpp_code
        )

