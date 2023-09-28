from dataclasses import dataclass
from .file_group_path import FileGroupPath 
from .component_type import ComponentType 
from ...path import AbsolutePath
from ..library import Library
from ...project import Project
from ...file_type import FileType
from .file_group_component import FileGroupComponent 
from ..cpp_code import CppCode 

@dataclass(frozen=True)
class FileGroup:
    logical_path: FileGroupPath['FileGroup']
    library: 'Library'
    cpp_type: FileType
    
    @property
    def project(self) -> 'Project':
        return self.cpp_code.project

    @property
    def cpp_code(self) -> 'CppCode':
        return self.library.cpp_code

    @property
    def source_extension(self) -> str:
        if self.cpp_type.implies(FileType.CUDA_CPP):
            return '.cu'
        elif self.cpp_type.implies(FileType.HIP_CPP):
            return '.cpp'
        elif self.cpp_type.implies(FileType.CPU_CPP):
            return '.cc'
        else:
            raise ValueError(f'Unhandled file type {self.cpp_type}')

    @property
    def source_path(self) -> AbsolutePath:
        return self.library.src_path / self.logical_path.raw.with_suffix('.cc')

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
    def create(cls, path: AbsolutePath, library: 'Library') -> 'FileGroup':
        return cls(
            logical_path=library.get_logical_path(path),
            library=library
        )

