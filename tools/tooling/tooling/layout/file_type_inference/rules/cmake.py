from tooling.layout.file_type_inference.rules.rule import Rule
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from tooling.layout.file_type_inference.file_attribute_set import FileAttributeSet
from tooling.layout.path import AbsolutePath, full_suffix
from typing import FrozenSet, Mapping

class CmakeRule(Rule):
    @property
    def outputs(self) -> FrozenSet[FileAttribute]:
        return frozenset({FileAttribute.CMAKE, FileAttribute.CMAKELISTS})

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return frozenset()

    def apply_to_path(self, p: AbsolutePath, attrs: Mapping[AbsolutePath, FileAttributeSet]) -> FrozenSet[FileAttribute]:
        if full_suffix(p) == '.cmake':
            return frozenset({FileAttribute.CMAKE})
        elif p.name == 'CMakeLists.txt':
            return frozenset({FileAttribute.CMAKELISTS})
        else:
            return frozenset()
