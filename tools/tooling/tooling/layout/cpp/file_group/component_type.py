from enum import Enum, auto
from tooling.layout.file_type import FileAttribute, FileAttributes
from typing import FrozenSet

class ComponentType(Enum):
    SOURCE = auto()
    TEST = auto()
    PRIVATE_HEADER = auto()
    PUBLIC_HEADER = auto()

    def attributes(self) -> FrozenSet[FileAttribute]:
        if self == ComponentType.PRIVATE_HEADER:
            return frozenset([FileAttribute.CPP_PRIVATE_HEADER])
        elif self == ComponentType.PUBLIC_HEADER:
            return frozenset([FileAttribute.CPP_PUBLIC_HEADER])
        elif self == ComponentType.SOURCE:
            return frozenset([FileAttribute.CPP_SOURCE])
        else:
            assert self == ComponentType.TEST
            return frozenset([FileAttribute.CPP_TEST])

    def is_header(self) -> bool:
        return self in [ComponentType.PUBLIC_HEADER, ComponentType.PRIVATE_HEADER]

    def is_implementation(self) -> bool:
        return not self.is_header()

    @staticmethod
    def all() -> FrozenSet['ComponentType']:
        return frozenset({
            ComponentType.SOURCE,
            ComponentType.TEST,
            ComponentType.PRIVATE_HEADER,
            ComponentType.PUBLIC_HEADER,
        })
