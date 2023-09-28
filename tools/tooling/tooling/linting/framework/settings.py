from dataclasses import dataclass
from tooling.linting.framework.method import Method

@dataclass
class Settings:
    method: Method
    force: bool
