from dataclasses import dataclass
from .path import LogicalPath, AbsolutePath
from .file_type import FileType
from typing import Set

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .library import Library

@dataclass(frozen=True)
class LogicalFile:
    logical_path: LogicalPath
    library: 'Library'
    
    @property
    def source_path(self) -> AbsolutePath:
        return self.library.src_path / self.logical_path.raw.with_suffix(FileType.SOURCE.extension())

    @property
    def test_path(self) -> AbsolutePath:
        return self.library.src_path / self.logical_path.raw.with_suffix(FileType.TEST.extension())

    @property
    def header_path(self) -> AbsolutePath:
        return self.library.include_path / self.logical_path.raw.with_suffix(FileType.HEADER.extension())

    @property
    def private_header_path(self) -> AbsolutePath:
        return self.library.src_path / self.logical_path.raw.with_suffix(FileType.HEADER.extension())

    def paths_for_file_type(self, file_type: FileType) -> Set[AbsolutePath]:
        if file_type == FileType.HEADER:
            return {self.header_path, self.private_header_path}
        elif file_type == FileType.TEST:
            return {self.test_path}
        else:
            assert file_type == FileType.SOURCE
            return {self.source_path}

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

