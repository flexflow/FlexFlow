from typing import Generic, Optional, TypeVar, Callable
from typing_extensions import TypeAlias
from abc import ABC, abstractmethod
from tooling.gh_mgmt.cache.core.previous_value_tag import PreviousValueTag
from tooling.gh_mgmt.issues.triage.config import get_beginning_of_time, get_cache, get_now
from sqlitedict import SqliteDict
from tooling.json import Json
import logging
from github import Github
from datetime import timedelta

_l = logging.getLogger(__name__)

T = TypeVar("T")
Lazy: TypeAlias = Callable[[], T]


class CacheEntry(ABC, Generic[T]):
    def _get_previous_value(self, cache: Optional[SqliteDict[str, Json]] = None) -> PreviousValueTag[Optional[T]]:
        if cache is None:
            cache = get_cache()
        cache_key = self._cache_entry_name()
        tag: PreviousValueTag[Optional[T]]
        if cache_key in cache:
            tag = PreviousValueTag.from_jsonable(self._from_jsonable, cache[cache_key])
            _l.debug(f"Cache key {cache_key} found in cache. Returning previous value tag: {tag}")
            return tag
        else:
            tag = PreviousValueTag(last_run=get_beginning_of_time(), last_value=self._default_value())
            _l.debug(f"Cache key {cache_key} not found in cache. Hallucinating previous value tag: {tag}")
            return tag

    def _save_in_cache(self, value: Lazy[T], cache: Optional[SqliteDict[str, Json]] = None) -> T:
        if cache is None:
            cache = get_cache()
        prev = self._get_previous_value(cache)

        time_alive = get_now() - prev.last_run
        cache_expired = time_alive > self._ttl()
        lookback_changed = self._history_dependent() and prev.last_beginning_of_time > get_beginning_of_time()
        if cache_expired or lookback_changed:
            cache_key = self._cache_entry_name()
            _l.info("Encountered cache miss on key: %s", cache_key)
            if cache_expired:
                _l.debug("Reason for miss: cache_expired (%s > %s)", time_alive, self._ttl())
            elif lookback_changed:
                _l.debug(
                    "Reason for miss: lookback_changed (%s > %s)", prev.last_beginning_of_time, get_beginning_of_time()
                )
            new_value = value()
            _l.debug("New value: %s", new_value)
            cache[cache_key] = PreviousValueTag(last_run=get_now(), last_value=new_value).as_jsonable(
                lambda t: self._to_jsonable(t)
            )
            cache.commit()
            return new_value
        else:
            assert prev.last_value is not None
            return prev.last_value

    def evaluate(self, g: Github) -> T:
        prev = self._get_previous_value()
        return self._post_hook(self._save_in_cache(lambda: self._update(self._fetch_new(g, prev), prev)))

    def _post_hook(self, t: T) -> T:
        return t

    @abstractmethod
    def _cache_entry_name(self) -> str:
        pass

    @abstractmethod
    def _update(self, new: T, prev: PreviousValueTag[Optional[T]]) -> T:
        pass

    @abstractmethod
    def _default_value(self) -> Optional[T]:
        pass

    @abstractmethod
    def _fetch_new(self, g: Github, prev: PreviousValueTag[Optional[T]]) -> T:
        pass

    @abstractmethod
    def _ttl(self) -> timedelta:
        pass

    @abstractmethod
    def _to_jsonable(self, value: T) -> Json:
        pass

    @abstractmethod
    def _from_jsonable(self, jsonable: Json) -> T:
        pass

    @abstractmethod
    def _history_dependent(self) -> bool:
        pass
