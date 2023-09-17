from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import Union, Iterator, Generator, TypeVar, Generic
import functools

@functools.cache
def cached_is_file(p: 'AbsolutePath'):
    return super(AbsolutePath, p).is_file()

@functools.cache
def cached_is_dir(p: 'AbsolutePath'):
    return super(AbsolutePath, p).is_dir()

@functools.cache
def cached_exists(p: 'AbsolutePath'):
    return super(AbsolutePath, p).exists()

class AbsolutePath(PosixPath):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    @classmethod
    def create(cls, base_path: Path, path: Path):
        return cls(path.absolute())

    def relative_to(self, path: 'AbsolutePath') -> Path:
        return super().relative_to(path)

    def is_relative_to(self, path: 'AbsolutePath') -> bool:
        return super().is_relative_to(path)

    def __truediv__(self, other: Union[Path, str]) -> 'AbsolutePath':
        assert isinstance(other, str) or (not other.is_absolute())
        return AbsolutePath(super().__truediv__(other))

    def is_file(self) -> bool:
        return cached_is_file(self)

    def is_dir(self) -> bool:
        return cached_is_dir(self)

    def exists(self) -> bool:
        return cached_exists(self)

    def iterdir(self) -> Generator['AbsolutePath', None, None]:
        for p in super().iterdir():
            yield AbsolutePath(p)

def with_all_suffixes_removed(p: Union[Path, AbsolutePath]):
    return p.with_name(p.name[:-len(''.join(p.suffixes))])

T = TypeVar('T')

@dataclass(frozen=True)
class LogicalPath(Generic[T]):
    raw: Path

    @classmethod
    def create(cls, base_path: Path, path: Path) -> 'LogicalPath':
        return LogicalPath(
            raw=with_all_suffixes_removed(path.relative_to(base_path))
        )
