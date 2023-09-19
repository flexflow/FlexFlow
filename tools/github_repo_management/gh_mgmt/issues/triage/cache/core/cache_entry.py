from typing import Generic, Optional, TypeVar, TypeAlias, Callable
from abc import ABC, abstractmethod
from .previous_value_tag import PreviousValueTag
from ...config.paths import get_cache
from ...config.time import get_maximum_lookback, get_beginning_of_time, get_now, ago
import logging
from github import Github
from datetime import timedelta
from ...data_model.json import Json

_l = logging.getLogger(__name__)

T = TypeVar("T")
Lazy: TypeAlias = Callable[[], T]


class CacheEntry(ABC, Generic[T]):
    def _get_previous_value(self, cache=None) -> PreviousValueTag[Optional[T]]:
        if cache is None:
            cache = get_cache()
        cache_key = self._cache_entry_name()
        if cache_key in cache:
            return PreviousValueTag.from_jsonable(self._from_jsonable, cache[cache_key])
        else:
            return PreviousValueTag(last_run=ago(get_maximum_lookback()), last_value=self._default_value())

    def _save_in_cache(self, value: Lazy[T], cache=None) -> T:
        if cache is None:
            cache = get_cache()
        prev = self._get_previous_value(cache)

        cache_expired = get_now() - prev.last_run > self._ttl()
        lookback_changed = self._history_dependent() and prev.last_beginning_of_time > get_beginning_of_time()
        if cache_expired or lookback_changed:
            cache_key = self._cache_entry_name()
            _l.info("Encountered cache miss on key: %s", cache_key)
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
