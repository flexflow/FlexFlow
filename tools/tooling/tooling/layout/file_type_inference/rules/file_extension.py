from tooling.layout.file_type_inference.rules.rule import Rule, HasExtension
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import FrozenSet

rules: FrozenSet[Rule] = frozenset({
    Rule(HasExtension('.cc'), FileAttribute.CPP_SOURCE),
    Rule(HasExtension('.cu'), FileAttribute.IS_CUDA_KERNEL),
    Rule(HasExtension('.h'), FileAttribute.HEADER),
    Rule(HasExtension('.py'), FileAttribute.PYTHON),
    Rule(HasExtension('.c'), FileAttribute.C),
    Rule(HasExtension('.test.cc'), FileAttribute.CPP_TEST),
})
