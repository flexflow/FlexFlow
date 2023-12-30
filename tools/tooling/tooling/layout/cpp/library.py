from dataclasses import dataclass
from typing import FrozenSet, Callable, cast, Optional
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from tooling.layout.file_type_inference.framework.inference_result import InferenceResult
from tooling.layout.cpp.file_group.file_group import FileGroup
from tooling.layout.path import AbsolutePath
from functools import lru_cache

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tooling.layout.cpp.cpp_code import CppCode
    from tooling.layout.project import Project


def include_path(library_root: AbsolutePath) -> AbsolutePath:
    return library_root / "include"


def src_path(library_root: AbsolutePath) -> AbsolutePath:
    return library_root / "src"


def cmake_file_path(library_root: AbsolutePath) -> AbsolutePath:
    return library_root / "CMakeLists.txt"


def is_library_root(path: AbsolutePath) -> bool:
    return all(
        [
            include_path(path).is_dir(),
            src_path(path).is_dir(),
            cmake_file_path(path).is_file(),
        ]
    )


def get_library_name(library_root: AbsolutePath) -> str:
    return ""


@dataclass(frozen=True)
class Library:
    name: str
    root_path: AbsolutePath
    cpp_code: "CppCode"

    @property
    def project(self) -> "Project":
        return self.cpp_code.project

    @property
    def file_types(self) -> InferenceResult:
        return self.project.file_types

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

    def is_valid_path(self, file_path: AbsolutePath) -> bool:
        assert file_path.is_relative_to(self.root_path)
        attrs = self.file_types.for_path(file_path)
        return attrs.issubset({FileAttribute.IN_CPP_LIBRARY, FileAttribute.CPP_LIBRARY_IS_VALID_FILE})

    @property
    def file_groups(self) -> FrozenSet[FileGroup]:
        return self._file_groups()

    @lru_cache()
    def _file_groups(self) -> FrozenSet[FileGroup]:
        results = set(
            FileGroup.try_create(path, library=self)
            for path in self.project.file_types.with_attr(FileAttribute.CPP_FILE_GROUP_MEMBER, within=self.root_path)
        )
        if None in results:
            results.remove(None)
        return frozenset(cast(FrozenSet[FileGroup], results))

    def files_satisfying(
        self, f: Callable[[AbsolutePath], bool], base_path: Optional[AbsolutePath] = None
    ) -> FrozenSet[AbsolutePath]:
        if base_path is None:
            base_path = self.root_path
        return self.project.files_satisfying(f, base_path)

    @classmethod
    def create(cls, root_path: AbsolutePath, cpp_code: "CppCode") -> "Library":
        assert is_library_root(root_path)
        return cls(name=get_library_name(root_path), root_path=root_path, cpp_code=cpp_code)
