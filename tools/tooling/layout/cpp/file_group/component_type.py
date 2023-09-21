from enum import Enum, auto
from ...file_type import FileType, FileTypes

class ComponentType(Enum):
    SOURCE = auto()
    TEST = auto()
    PRIVATE_HEADER = auto()
    PUBLIC_HEADER = auto()

    def file_type(self, cpp_class: FileType) -> FileTypes:
        if self == ComponentType.PRIVATE_HEADER:
            return FileTypes(FileType._to_cpp_type(FileType.CPP_PRIVATE_HEADER, cpp_class))
        elif self == ComponentType.SOURCE:
            return FileType.SOURCE
        else:
            assert self == ComponentType.TEST
            return FileType.TEST


    def is_header(self) -> bool:
        return self in [ComponentType.PUBLIC_HEADER, ComponentType.PRIVATE_HEADER]

    def is_implementation(self) -> bool:
        return not self.is_header()

