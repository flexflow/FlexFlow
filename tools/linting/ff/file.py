from dataclasses import dataclass
from .path import LogicalPath, AbsolutePath
from .file_type import FileType
from typing import Set
from enum import Enum, auto

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .library import Library
    from .project import Project

class ComponentType(Enum):
    SOURCE = auto()
    TEST = auto()
    PRIVATE_HEADER = auto()
    PUBLIC_HEADER = auto()

    @property
    def file_type(self) -> FileType:
        if self.is_header():
            return FileType.HEADER
        elif self == ComponentType.SOURCE:
            return FileType.SOURCE
        else:
            assert self == ComponentType.TEST
            return FileType.TEST


    def is_header(self) -> bool:
        return self in [ComponentType.PUBLIC_HEADER, ComponentType.PRIVATE_HEADER]

    def is_implementation(self) -> bool:
        return not self.is_header()

@dataclass(frozen=True)
class LogicalFileComponent:
    logical_file: 'LogicalFile'
    component_type: ComponentType

    def exists(self) -> bool:
        return self.path.is_file()

    @property
    def extension(self) -> str:
        return self.file_type.extension()

    @property
    def file_type(self) -> FileType:
        return self.component_type.file_type 

    @property
    def path(self) -> AbsolutePath:
        return self.logical_file.path_of_component(self.component_type)

    @property
    def file_logical_path(self) -> LogicalPath['LogicalFile']:
        return self.logical_file.logical_path 

    @property
    def library_logical_path(self) -> LogicalPath['Library']:
        return self.logical_file.library.get_library_relative_logical_path(self.path)

    @property
    def project_logical_path(self) -> LogicalPath:
        return self.logical_file.project.get_project_relative_logical_path(self.path)

@dataclass(frozen=True)
class LogicalFile:
    logical_path: LogicalPath['LogicalFile']
    library: 'Library'
    
    @property
    def project(self) -> 'Project':
        return self.library.project

    @property
    def source_path(self) -> AbsolutePath:
        return self.library.src_path / self.logical_path.raw.with_suffix(FileType.SOURCE.extension())

    @property
    def test_path(self) -> AbsolutePath:
        return self.library.src_path / self.logical_path.raw.with_suffix(FileType.TEST.extension())

    @property
    def public_header_path(self) -> AbsolutePath:
        return self.library.include_path / self.logical_path.raw.with_suffix(FileType.HEADER.extension())

    @property
    def private_header_path(self) -> AbsolutePath:
        return self.library.src_path / self.logical_path.raw.with_suffix(FileType.HEADER.extension())

    def paths_for_file_type(self, file_type: FileType) -> Set[AbsolutePath]:
        if file_type == FileType.HEADER:
            return {self.public_header_path, self.private_header_path}
        elif file_type == FileType.TEST:
            return {self.test_path}
        else:
            assert file_type == FileType.SOURCE
            return {self.source_path}

    @property
    def public_header(self) -> LogicalFileComponent:
        return self.get_component(ComponentType.PUBLIC_HEADER)

    @property
    def private_header(self) -> LogicalFileComponent:
        return self.get_component(ComponentType.PRIVATE_HEADER)

    @property
    def source_file(self) -> LogicalFileComponent:
        return self.get_component(ComponentType.SOURCE)

    @property
    def test_file(self) -> LogicalFileComponent:
        return self.get_component(ComponentType.TEST)

    def get_component(self, component_type: ComponentType) -> LogicalFileComponent:
        return LogicalFileComponent(
            logical_file=self,
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

    def paths_for_file_types(self, file_types: Set[FileType]) -> Set[AbsolutePath]:
        result: Set[AbsolutePath] = set()
        for file_type in file_types:
            result |= self.paths_for_file_type(file_type)
        return result

    @classmethod
    def create(cls, path: AbsolutePath, library: 'Library') -> 'LogicalFile':
        return cls(
            logical_path=library.get_logical_path(path),
            library=library
        )

