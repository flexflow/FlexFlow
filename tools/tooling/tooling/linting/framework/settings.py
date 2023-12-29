from dataclasses import dataclass
from tooling.linting.framework.method import Method


@dataclass
class Settings:
    force: bool
