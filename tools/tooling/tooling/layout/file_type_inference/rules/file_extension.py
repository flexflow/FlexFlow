from tooling.layout.file_type_inference.rules.rule import Rule
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from tooling.layout.file_type_inference.file_attribute_set import FileAttributeSet
from tooling.layout.path import AbsolutePath, full_suffix
from typing import FrozenSet, Mapping

class FileExtensionRule(Rule):
    @property
    def outputs(self) -> FrozenSet[FileAttribute]:
        return frozenset({
            FileAttribute.CPP_SOURCE,
            FileAttribute.CPP_TEST,
            FileAttribute.IS_CUDA_KERNEL,
            FileAttribute.IS_HIP_KERNEL,
            FileAttribute.HEADER,
            FileAttribute.PYTHON,
            FileAttribute.C,
        })

    @property
    def inputs(self) -> FrozenSet[FileAttribute]:
        return frozenset()

    @property
    def _suffix_dict(self) -> Mapping[str, FrozenSet[FileAttribute]]:
        _d = {
            '.cc' : FileAttribute.CPP_SOURCE,
            '.cu' : FileAttribute.IS_CUDA_KERNEL,
            '.h'  : FileAttribute.HEADER,
            '.py' : FileAttribute.PYTHON,
            '.c'  : FileAttribute.C,
            '.test.cc' : FileAttribute.CPP_TEST,
        }
        return {k : frozenset({v}) for k, v in _d.items()}

    def apply_to_path(self, p: AbsolutePath, attrs: Mapping[AbsolutePath, FileAttributeSet]) -> FrozenSet[FileAttribute]:
        suffix = full_suffix(p)
        return self._suffix_dict.get(suffix, frozenset())
