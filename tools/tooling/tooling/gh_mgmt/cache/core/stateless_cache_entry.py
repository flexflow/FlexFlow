from typing import Optional, TypeVar
from tooling.gh_mgmt.cache.core.cache_entry import CacheEntry
from tooling.gh_mgmt.cache.core.previous_value_tag import PreviousValueTag
from abc import abstractmethod
from github import Github

T = TypeVar("T")


class StatelessCacheEntry(CacheEntry[T]):
    def _update(self, new: T, prev: PreviousValueTag[Optional[T]]) -> T:
        return new

    def _default_value(self) -> Optional[T]:
        return None

    def _fetch_new(self, g: Github, prev: PreviousValueTag[Optional[T]]) -> T:
        return self._fetch_stateless(g)

    @abstractmethod
    def _fetch_stateless(self, g: Github) -> T:
        pass

    def _history_dependent(self) -> bool:
        return False
