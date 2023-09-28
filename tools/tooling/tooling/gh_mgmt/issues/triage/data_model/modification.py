from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, order=True)
class Modification:
    updated_at: datetime
    user: str

    def as_jsonable(self) -> Any:
        return {"updated_at": self.updated_at.isoformat(), "user": self.user}

    @classmethod
    def from_jsonable(cls, jsonable: Any) -> "Modification":
        return cls(updated_at=datetime.fromisoformat(jsonable["updated_at"]), user=jsonable["user"])
