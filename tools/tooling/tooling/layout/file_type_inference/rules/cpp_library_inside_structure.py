from tooling.layout.file_type_inference.rules.rule import Rule
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from tooling.layout.file_type_inference.file_attribute_set import FileAttributeSet
from tooling.layout.path import AbsolutePath
from typing import FrozenSet, Mapping

class CppLibraryInsideStructureRule(Rule):
    @property
    def outputs(self) -> FrozenSet[FileAttribute]:
        return frozenset({FileAttribute.CPP_LIBRARY_IN_INCLUDE, FileAttribute.CPP_LIBRARY_IN_SRC})

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return frozenset({FileAttribute.CPP_LIBRARY_SRC_DIR, FileAttribute.CPP_LIBRARY_INCLUDE_DIR})

    def apply_to_path(self, p: AbsolutePath, attrs: Mapping[AbsolutePath, FileAttributeSet]) -> FrozenSet[FileAttribute]:
        if any(attrs[parent].implies(FileAttribute.CPP_LIBRARY_SRC_DIR) for parent in p.parents):
            return frozenset({FileAttribute.CPP_LIBRARY_IN_SRC})
        elif any(attrs[parent].implies(FileAttribute.CPP_LIBRARY_INCLUDE_DIR) for parent in p.parents):
            return frozenset({FileAttribute.CPP_LIBRARY_IN_INCLUDE})
        return frozenset()
