from tooling.layout.file_type_inference.rules.rule import Rule
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from tooling.layout.file_type_inference.file_attribute_set import FileAttributeSet
from tooling.layout.path import AbsolutePath
from typing import FrozenSet, Mapping

class CppLibraryRule(Rule):
    @property
    def outputs(self) -> FrozenSet[FileAttribute]:
        return frozenset({FileAttribute.CPP_LIBRARY})

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return frozenset({FileAttribute.CMAKELISTS})

    def children(self, p: AbsolutePath) -> Iterator[AbsolutePath]:
        for child in p.iterdir():
            yield child
            if child.is_dir():
                yield from self.children(child)

    def apply_to_path(self, p: AbsolutePath, attrs: Mapping[AbsolutePath, FileAttributeSet]) -> FrozenSet[FileAttribute]:
        if all([
            p.is_dir(), 
            (p / 'src').is_dir(), 
            (p / 'include').is_dir(), 
            attrs[p / 'CMakeLists.txt'].implies(FileAttribute.CMAKELISTS), 
            not any(attrs[child].implies(FileAttribute.CPP_LIBRARY) for child in self.children(p)),
                not any(attrs[parent].implies(FileAttribute.CPP_LIBRARY) for parent in p.parents)]):
            return frozenset({FileAttribute.CPP_LIBRARY})
        else:
            return frozenset()

