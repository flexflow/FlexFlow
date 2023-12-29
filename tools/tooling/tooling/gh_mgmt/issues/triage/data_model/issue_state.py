from enum import Enum


class IssueState(Enum):
    open = "open"
    closed = "closed"
    all = "all"

    def __str__(self) -> str:
        return self.value
