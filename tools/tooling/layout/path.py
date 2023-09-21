from pathlib import Path, PosixPath, PurePath, PurePosixPath
from os import PathLike
from typing import Union, Generator, cast, Any
import functools

@functools.lru_cache()
def cached_is_file(p: 'AbsolutePath') -> bool:
    return super(AbsolutePath, p).is_file()

@functools.lru_cache()
def cached_is_dir(p: 'AbsolutePath') -> bool:
    return super(AbsolutePath, p).is_dir()

@functools.lru_cache()
def cached_exists(p: 'AbsolutePath') -> bool:
    return super(AbsolutePath, p).exists()

class AbsolutePath(Path):
    @classmethod
    def create(cls, base_path: Path, path: Path) -> 'AbsolutePath':
        return cls(path.absolute())

    def relative_to(self, other: Union[str, PathLike[str]]) -> Path: # type: ignore
        return super().relative_to(other)

    def __truediv__(self, other: Union[str, PathLike[str]]) -> 'AbsolutePath':
        _other = Path(other)
        assert not _other.is_absolute()
        return AbsolutePath(super().__truediv__(cast(Any, other)))

    def is_file(self) -> bool:
        return cached_is_file(self)

    def is_dir(self) -> bool:
        return cached_is_dir(self)

    def exists(self) -> bool:
        return cached_exists(self)

    def iterdir(self) -> Generator['AbsolutePath', None, None]:
        for p in super().iterdir():
            yield AbsolutePath(p)

def with_all_suffixes_removed(p: Union[Path, AbsolutePath]) -> Path:
    return p.with_name(p.name[:-len(''.join(p.suffixes))])
