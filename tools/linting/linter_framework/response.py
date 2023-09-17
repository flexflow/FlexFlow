from dataclasses import dataclass
import json
import sys
from typing import Any, Optional

@dataclass(frozen=True)
class Response:
    return_code: int
    data: Any = None
    json_data: Any = None
    message: Optional[str] = None
    @staticmethod
    def success(**kwargs) -> 'Response':
        return Response(
            return_code=0,
            **kwargs,
        )

    @staticmethod
    def failure(**kwargs) -> 'Response':
        return Response(
            return_code=1,
            **kwargs,
        )

    def show(self) -> None:
        if self.message is not None:
            print(self.message, file=sys.stderr)
        elif self.json_data is not None:
            print(json.dumps(self.json_data, indent=2), file=sys.stdout)

