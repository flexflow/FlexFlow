from tooling.layout.path import AbsolutePath
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import (
    Dict,
    Set,
    DefaultDict,
    Optional,
    FrozenSet,
    Iterable,
    Any,
    Callable,
)
from collections import defaultdict
import logging

_l = logging.getLogger(__name__)


class InferenceResult:
    def __init__(
        self, 
        attrs: Dict[AbsolutePath, FrozenSet[FileAttribute]], 
        get_saved: Callable[[FileAttribute], Any],
    ) -> None:
        self._attrs = attrs
        self._reverse_attrs: DefaultDict[FileAttribute, Set[AbsolutePath]] = defaultdict(set)
        self._get_saved = get_saved
        for k, v in attrs.items():
            for a in v:
                self._reverse_attrs[a].add(k)

    def get_saved(self, attr: FileAttribute) -> Any:
        return self._get_saved(attr)

    def for_path(self, p: AbsolutePath) -> FrozenSet[FileAttribute]:
        return frozenset(self._attrs[p])

    def with_attr(self, attr: FileAttribute, within: Optional[AbsolutePath] = None) -> FrozenSet[AbsolutePath]:
        if within is None:
            return frozenset(self._reverse_attrs[attr])
        else:
            return frozenset(p for p in self._reverse_attrs[attr] if p.is_relative_to(within))

    def without_attr(self, attr: FileAttribute, within: Optional[AbsolutePath] = None) -> FrozenSet[AbsolutePath]:
        return self._all_paths(within) - self._reverse_attrs[attr]

    def _all_paths(self, within: Optional[AbsolutePath] = None) -> FrozenSet[AbsolutePath]:
        if within is None:
            return frozenset(self._attrs.keys())
        else:
            return frozenset(p for p in self._attrs.keys() if p.is_relative_to(within))

    def without_all_of_attrs(
        self, attrs: Iterable[FileAttribute], within: Optional[AbsolutePath] = None
    ) -> FrozenSet[AbsolutePath]:
        return self._all_paths(within) - self.with_all_of_attrs(attrs, within=within)

    def without_any_of_attrss(
        self, attrs: Iterable[FileAttribute], within: Optional[AbsolutePath] = None
    ) -> FrozenSet[AbsolutePath]:
        return self._all_paths(within) - self.with_any_of_attrs(attrs, within=within)

    def with_all_of_attrs(
        self, attrs: Iterable[FileAttribute], within: Optional[AbsolutePath] = None
    ) -> FrozenSet[AbsolutePath]:
        result: Optional[FrozenSet[AbsolutePath]] = None
        for at in attrs:
            if result is None:
                result = self.with_attr(at, within=within)
            else:
                result &= self.with_attr(at, within=within)

        if result is None:
            raise ValueError("Cannot call with_all_of_attrs on empty attr set")
        else:
            return result

    def with_any_of_attrs(
        self, attrs: Iterable[FileAttribute], within: Optional[AbsolutePath] = None
    ) -> FrozenSet[AbsolutePath]:
        result: Optional[FrozenSet[AbsolutePath]] = None
        for at in attrs:
            if result is None:
                result = self.with_attr(at, within=within)
            else:
                result |= self.with_attr(at, within=within)

        if result is None:
            raise ValueError("Cannot call with_any_of_attrs on empty attr set")
        else:
            return result
