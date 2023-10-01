from tooling.layout.file_type_inference.rules.rule import Rule
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from tooling.layout.file_type_inference.file_attribute_set import FileAttributeSet
from tooling.layout.path import AbsolutePath
from typing import FrozenSet, Mapping

class PrivateHeaderRule(Rule):
    @property
    def outputs(self) -> FrozenSet[FileAttribute]:
        return frozenset({
            FileAttribute.CPP_FWDING_HEADER
        })

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return frozenset({FileAttribute.HEADER})

    def apply_to_path(self, p: AbsolutePath, attrs: Mapping[AbsolutePath, FileAttributeSet]) -> FrozenSet[FileAttribute]:
        if attrs[p].implies(FileAttribute.HEADER):
            with p.open('r') as f:
                lines = f.read().splitlines()
            if all(line.startswith('#include') for line in lines):
                return frozenset({FileAttribute.CPP_FWDING_HEADER})
        return frozenset()
