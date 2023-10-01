from abc import ABC, abstractmethod, abstractproperty
from tooling.layout.path import AbsolutePath
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from tooling.layout.file_type_inference.file_attribute_set import FileAttributeSet
from typing import FrozenSet, Mapping

class Rule(ABC):
    @abstractproperty
    def outputs(self) -> FrozenSet[FileAttribute]:
        ...

    @abstractproperty
    def inputs(self) -> FrozenSet[FileAttribute]:
        ...

    @abstractmethod
    def apply_to_path(self, p: AbsolutePath, attrs: Mapping[AbsolutePath, FileAttributeSet]) -> FrozenSet[FileAttribute]:
        ...

