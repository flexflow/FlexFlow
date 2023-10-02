from dataclasses import dataclass
from typing import Optional, FrozenSet
from tooling.layout.cpp.file_group.component_type import ComponentType 
from tooling.layout.path import AbsolutePath, with_all_suffixes_removed, full_suffix 
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from tooling.layout.cpp.file_group.file_group_component import FileGroupComponent 
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tooling.layout.cpp.cpp_code import CppCode
    from tooling.layout.cpp.library import Library
    from tooling.layout.project import Project


@dataclass(frozen=True)
class FileGroup:
    library: 'Library'
    _logical_path: Path
    
    @property
    def project(self) -> 'Project':
        return self.cpp_code.project

    @property
    def cpp_code(self) -> 'CppCode':
        return self.library.cpp_code

    @property
    def source_extension(self) -> str:
        return '.cc'

    @property
    def test_extension(self) -> str:
        return '.test' + self.source_extension

    @property
    def header_extension(self) -> str:
        return '.h'
    #property

    @property
    def source_path(self) -> AbsolutePath:
        return self.library.src_path / self._logical_path.with_suffix(self.source_extension)

    @property
    def test_path(self) -> AbsolutePath:
        return self.library.src_path / self._logical_path.with_suffix(self.test_extension)

    @property
    def public_header_path(self) -> AbsolutePath:
        return self.library.include_path / self._logical_path.with_suffix(self.header_extension)

    @property
    def private_header_path(self) -> AbsolutePath:
        return self.library.src_path / self._logical_path.with_suffix(self.header_extension)

    @property
    def public_header(self) -> FileGroupComponent:
        return self.get_component(ComponentType.PUBLIC_HEADER)

    @property
    def private_header(self) -> FileGroupComponent:
        return self.get_component(ComponentType.PRIVATE_HEADER)

    @property
    def source_file(self) -> FileGroupComponent:
        return self.get_component(ComponentType.SOURCE)

    @property
    def test_file(self) -> FileGroupComponent:
        return self.get_component(ComponentType.TEST)

    def get_component(self, component_type: ComponentType) -> FileGroupComponent:
        return FileGroupComponent(
            file_group=self,
            component_type=component_type
        )

    def path_of_component(self, component_type: ComponentType) -> AbsolutePath:
        if component_type == ComponentType.PUBLIC_HEADER:
            return self.public_header_path
        elif component_type == ComponentType.PRIVATE_HEADER:
            return self.private_header_path
        elif component_type == ComponentType.SOURCE:
            return self.source_path
        else:
            assert component_type == ComponentType.TEST
            return self.test_path

    @property
    def all_component_paths(self) -> FrozenSet[AbsolutePath]:
        return frozenset(self.path_of_component(ct) for ct in ComponentType.all())

    @property
    def existing_components(self) -> FrozenSet[AbsolutePath]:
        return frozenset(p for p in self.all_component_paths if p.is_file())

    @property
    def missing_components(self) -> FrozenSet[AbsolutePath]:
        return self.all_component_paths - self.existing_components

    @classmethod
    def try_create(cls, path: AbsolutePath, library: 'Library') -> Optional['FileGroup']:
        if FileAttribute.CPP_FILE_GROUP_MEMBER in library.file_types.for_path(path):
            base_path = next(parent for parent in path.parents if FileAttribute.CPP_FILE_GROUP_BASE in library.file_types.for_path(parent))
            return FileGroup(
                library=library, _logical_path=with_all_suffixes_removed(path.relative_to(base_path))
            )
        else:
            return None
