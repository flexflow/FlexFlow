from tooling.layout.file_type_inference.rules.rule import Rule
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from tooling.layout.file_type_inference.file_attribute_set import FileAttributeSet
from tooling.layout.path import AbsolutePath
from typing import FrozenSet, Mapping

class PrivateHeaderRule(Rule):
    @property
    def outputs(self) -> FrozenSet[FileAttribute]:
        return frozenset({
            FileAttribute.CPP_PRIVATE_HEADER
        })

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return frozenset({FileAttribute.CPP_LIBRARY_IN_SRC})

    def apply_to_path(self, p: AbsolutePath, attrs: Mapping[AbsolutePath, FileAttributeSet]) -> FrozenSet[FileAttribute]:
        if attrs[p].implies_all_of([FileAttribute.HEADER, FileAttribute.CPP_LIBRARY_IN_SRC]):
            return frozenset({FileAttribute.CPP_PRIVATE_HEADER})
        return frozenset()
