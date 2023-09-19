from typing import Generic, Callable, Any, TypeVar
from datetime import datetime
from dataclasses import dataclass, field
from ...config.time import get_beginning_of_time

T = TypeVar("T")


@dataclass(frozen=True)
class PreviousValueTag(Generic[T]):
    last_run: datetime
    last_value: T
    last_beginning_of_time: datetime = field(default_factory=get_beginning_of_time)

    def as_jsonable(self, value_as_jsonable: Callable[[T], Any]) -> Any:
        return {
            "last_run": self.last_run.isoformat(),
            "last_value": value_as_jsonable(self.last_value),
            "last_beginning_of_time": self.last_beginning_of_time.isoformat(),
        }

    @classmethod
    def from_jsonable(cls, value_from_jsonable: Callable[[Any], T], jsonable: Any) -> "PreviousValueTag[T]":
        assert isinstance(jsonable, dict), type(jsonable)
        return cls(
            last_run=datetime.fromisoformat(jsonable["last_run"]),
            last_value=value_from_jsonable(jsonable["last_value"]),
            last_beginning_of_time=datetime.fromisoformat(jsonable["last_beginning_of_time"]),
        )
