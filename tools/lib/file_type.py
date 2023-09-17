from enum import Enum
from pathlib import Path
from typing import Optional, Union, Set, Iterator
from .path import AbsolutePath
import functools

class FileType(Enum):
    SOURCE = 'source'
    TEST = 'test'
    HEADER = 'header'

    def __str__(self):
        return self.name

    @staticmethod
    def all() -> frozenset['FileType']:
        return frozenset([FileType.HEADER, FileType.SOURCE, FileType.TEST])

    def others(self) -> frozenset['FileType']:
        return frozenset(FileType.all()) - {self}

    def extension(self):
        if self == FileType.HEADER:
            return '.h'
        elif self == FileType.SOURCE:
            return '.cc'
        else:
            assert self == FileType.TEST
            return '.test.cc'

def file_type_of_path(path: Union[Path, AbsolutePath]) -> Optional[FileType]:
    for file_type in FileType.all():
        if has_correct_extension(file_type, path):
            return file_type
    return None

def has_correct_extension(file_type: FileType, p: Union[Path, AbsolutePath]) -> bool:
    return ''.join(p.suffixes) == file_type.extension()

@functools.cache
def find_in_path_with_file_type(
    path: AbsolutePath,
    file_types: frozenset[FileType],
    skip_directories: Optional[frozenset[AbsolutePath]] = None
) -> frozenset[AbsolutePath]:
    return frozenset(find_in_path_with_file_type_iter(path, file_types, skip_directories))

def find_in_path_with_file_type_iter(
    path: AbsolutePath, 
    file_types: frozenset[FileType], 
    skip_directories: Optional[frozenset[AbsolutePath]] = None
) -> Iterator[AbsolutePath]:

    if skip_directories is None:
        skip_directories = frozenset()

    def _recurse(p):
        if p in skip_directories:
            return
        if p.is_dir():
            for child in p.iterdir():
                yield from _recurse(child)
        elif p.is_file() and file_type_of_path(p) in file_types:
            yield p
        
    yield from _recurse(path)
    
