from dataclasses import dataclass

from typing import TYPE_CHECKING
from .component_type import ComponentType
from ...file_type import FileType
from ...path import AbsolutePath
if TYPE_CHECKING:
    from .file_group import FileGroup

@dataclass(frozen=True)
class FileGroupComponent:
    file_group: 'FileGroup'
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

