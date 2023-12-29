from dataclasses import dataclass
from tooling.json import Json
from typing import Union, Optional


@dataclass(frozen=True)
class CheckResponse:
    num_errors: int
    json_data: Json = None

    def as_json(self) -> Json:
        return {
            "num_errors": self.num_errors,
            "json_data": self.json_data,
        }


@dataclass(frozen=True)
class FixResponse:
    did_succeed: bool
    num_fixes: Optional[int] = None
    message: Optional[str] = None
    json_data: Json = None

    def as_json(self) -> Json:
        return {
            "did_succeed": self.did_succeed,
            "num_fixes": self.num_fixes,
            "message": self.message,
            "json_data": self.json_data,
        }


@dataclass(frozen=True)
class ErrorResponse:
    message: str

    def as_json(self) -> Json:
        return {"message": self.message}


Response = Union[CheckResponse, FixResponse, ErrorResponse]


def did_succeed(r: Response) -> bool:
    if isinstance(r, CheckResponse):
        return r.num_errors == 0
    elif isinstance(r, FixResponse):
        return r.did_succeed
    else:
        assert isinstance(r, ErrorResponse)
        return False
