from dataclasses import dataclass
from tooling.json import Json
from typing import Union

@dataclass(frozen=True)
class CheckResponse:
    num_errors: int
    json_data: Json

@dataclass(frozen=True)
class FixResponse:
    num_fixes: int
    json_data: Json

Response = Union[CheckResponse, FixResponse]
