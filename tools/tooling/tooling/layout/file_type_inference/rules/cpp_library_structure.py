from tooling.layout.file_type_inference.rules.rule import Rule
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from tooling.layout.file_type_inference.file_attribute_set import FileAttributeSet
from tooling.layout.path import AbsolutePath
from typing import FrozenSet, Mapping

class CppLibraryStructureRule(Rule):
    @property
    def outputs(self) -> FrozenSet[FileAttribute]:
        return frozenset({FileAttribute.CPP_LIBRARY_SRC_DIR, FileAttribute.CPP_LIBRARY_INCLUDE_DIR})

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return frozenset({FileAttribute.CPP_LIBRARY})

    def apply_to_path(self, p: AbsolutePath, attrs: Mapping[AbsolutePath, FileAttributeSet]) -> FrozenSet[FileAttribute]:
        if p.is_dir() and attrs[p.parent].implies(FileAttribute.CPP_LIBRARY):
            if p.name == 'src':
                return frozenset({FileAttribute.CPP_LIBRARY_SRC_DIR})
            elif p.name == 'include':
                return frozenset({FileAttribute.CPP_LIBRARY_INCLUDE_DIR})
        return frozenset()
