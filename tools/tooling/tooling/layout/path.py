from pathlib import Path, PosixPath
from os import PathLike
from typing import Union, Generator, cast, Any, Optional, Iterator, Callable
import functools


@functools.lru_cache(maxsize=None)
def cached_is_file(p: "AbsolutePath") -> bool:
    return super(AbsolutePath, p).is_file()


@functools.lru_cache(maxsize=None)
def cached_is_dir(p: "AbsolutePath") -> bool:
    return super(AbsolutePath, p).is_dir()


@functools.lru_cache(maxsize=None)
def cached_exists(p: "AbsolutePath") -> bool:
    return super(AbsolutePath, p).exists()


class AbsolutePath(PosixPath):
    @classmethod
    def create(cls, path: Path, base_path: Optional[Path] = None) -> "AbsolutePath":
        if base_path is None:
            base_path = Path.cwd()
        return cls(base_path / path)

    def relative_to(self, other: Union[str, PathLike[str]]) -> Path:  # type: ignore
        return super().relative_to(other)

    def __truediv__(self, other: Union[str, PathLike[str]]) -> "AbsolutePath":
        _other = Path(other)
        assert not _other.is_absolute()
        return AbsolutePath(super().__truediv__(cast(Any, other)))

    def is_file(self) -> bool:
        return cached_is_file(self)

    def is_dir(self) -> bool:
        return cached_is_dir(self)

    def exists(self) -> bool:
        return cached_exists(self)

    def iterdir(self) -> Generator["AbsolutePath", None, None]:
        for p in super().iterdir():
            yield AbsolutePath(p)

    def is_relative_to(self, other: "AbsolutePath") -> bool:
        try:
            self.relative_to(other)
            return True
        except ValueError:
            return False


def with_all_suffixes_removed(p: Union[Path, AbsolutePath]) -> Path:
    return p.with_name(p.name[: -len("".join(p.suffixes))])


def full_suffix(p: Union[Path, AbsolutePath]) -> str:
    return "".join(p.suffixes)


def children(
    p: AbsolutePath, traverse_children: Optional[Callable[[AbsolutePath], bool]] = None
) -> Iterator[AbsolutePath]:
    for child in p.iterdir():
        yield child
        if child.is_dir() and (traverse_children is None or traverse_children(child)):
            yield from children(child, traverse_children=traverse_children)
