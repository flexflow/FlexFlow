from tooling.layout.file_type_inference.rules.rule import Rule, HasExtension
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import FrozenSet

rules: FrozenSet[Rule] = frozenset({
    Rule(
        'extension.cc',
        HasExtension('.cc'), 
        FileAttribute.CPP_SOURCE
    ),
    Rule(
        'extension.cu',
        HasExtension('.cu'), 
        FileAttribute.IS_CUDA_KERNEL
    ),
    Rule(
        'extension.h',
        HasExtension('.h'), 
        FileAttribute.HEADER
    ),
    Rule(
        'extension.py',
        HasExtension('.py'), 
        FileAttribute.PYTHON
    ),
    Rule(
        'extension.c',
        HasExtension('.c'), 
        FileAttribute.C
    ),
    Rule(
        'extension.test.cc',
        HasExtension('.test.cc'), 
        FileAttribute.CPP_TEST
    ),
})
