# from enum import Enum
# from pathlib import Path
# from typing import Optional, Union, Iterator, FrozenSet
# from .path import AbsolutePath
# import functools

# def file_type_of_path(path: Union[Path, AbsolutePath]) -> Optional[FileType]:
#     for file_type in FileType.all():
#         if has_correct_extension(file_type, path):
#             return file_type
#     return None

# def has_correct_extension(file_type: FileType, p: Union[Path, AbsolutePath]) -> bool:
#     return ''.join(p.suffixes) == file_type.extension()

# @functools.lru_cache()
# def find_in_path_with_file_type(
#     path: AbsolutePath, 
#     file_types: FrozenSet[FileType], 
#     skip_directories: FrozenSet[AbsolutePath] = frozenset()
# ) -> FrozenSet[AbsolutePath]:

#     def _recurse(p: AbsolutePath) -> Iterator[AbsolutePath]:
#         if p in skip_directories:
#             return
#         if p.is_dir():
#             for child in p.iterdir():
#                 yield from _recurse(child)
#         elif p.is_file() and file_type_of_path(p) in file_types:
#             yield p
        
#     return frozenset(_recurse(path))
    
