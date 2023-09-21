from pathlib import Path
from typing import Generic, TypeVar
from dataclasses import dataclass
from ...path import with_all_suffixes_removed 

T = TypeVar('T')

@dataclass(frozen=True)
class FileGroupPath(Generic[T]):
    raw: Path

    @classmethod
    def create(cls, base_path: Path, path: Path) -> 'FileGroupPath[T]':
        return FileGroupPath(
            raw=with_all_suffixes_removed(path.relative_to(base_path))
        )
