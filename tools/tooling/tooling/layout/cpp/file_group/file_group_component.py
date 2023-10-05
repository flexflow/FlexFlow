from dataclasses import dataclass

from typing import TYPE_CHECKING
from tooling.layout.cpp.file_group.component_type import ComponentType 
from tooling.layout.path import AbsolutePath

if TYPE_CHECKING:
    from tooling.layout.cpp.file_group.file_group import FileGroup 
    from tooling.layout.cpp.library import Library
    from tooling.layout.cpp.cpp_code import CppCode
    from tooling.layout.project import Project

@dataclass(frozen=True)
class FileGroupComponent:
    file_group: 'FileGroup'
    component_type: ComponentType

    def exists(self) -> bool:
        return self.path.is_file()

    @property
    def path(self) -> AbsolutePath:
        return self.file_group.path_of_component(self.component_type)

    @property
    def library(self) -> 'Library':
        return self.file_group.library

    @property
    def cpp_code(self) -> 'CppCode':
        return self.library.cpp_code 

    @property
    def project(self) -> 'Project':
        return self.cpp_code.project
