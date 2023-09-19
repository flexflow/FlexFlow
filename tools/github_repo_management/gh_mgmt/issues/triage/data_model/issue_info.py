from dataclasses import dataclass
from typing import FrozenSet, Optional, Any
from datetime import datetime
from .issue_state import IssueState
from .json import Json


@dataclass(frozen=True)
class IssueInfo:
    assignees: FrozenSet[str]
    state: IssueState
    pull_request: Optional[int]
    closed_at: Optional[datetime]

    def as_jsonable(self) -> Json:
        return {
            "assignees": list(sorted(self.assignees)),
            "state": self.state.value,
            "pull_request": self.pull_request,
            "closed_at": self.closed_at.isoformat() if self.closed_at is not None else None,
        }

    @classmethod
    def from_jsonable(cls, jsonable: Any) -> "IssueInfo":
        return cls(
            assignees=frozenset(jsonable["assignees"]),
            state=IssueState(jsonable["state"]),
            pull_request=jsonable["pull_request"],
            closed_at=datetime.fromisoformat(jsonable["closed_at"]) if jsonable["closed_at"] is not None else None,
        )

    def closed_after(self, d: datetime) -> Optional[bool]:
        if self.closed_at is None:
            return None
        else:
            assert self.state == IssueState.closed
            return self.closed_at >= d

    def closed_before(self, d: datetime) -> Optional[bool]:
        if self.closed_at is None:
            return None
        else:
            assert self.state == IssueState.closed
            return self.closed_at <= d

    @staticmethod
    def id_from_url(url: str) -> int:
        return int(url.split("/")[-1])
