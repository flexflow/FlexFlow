from tooling.layout.file_type_inference.rules.rule import Rule, HasExtension, exclude_blacklisted
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import FrozenSet

rules: FrozenSet[Rule] = frozenset({
    Rule(
        'extension.cc',
        exclude_blacklisted(HasExtension('.cc')), 
        FileAttribute.CPP_SOURCE
    ),
    Rule(
        'extension.cu',
        exclude_blacklisted(HasExtension('.cu')), 
        FileAttribute.IS_CUDA_KERNEL
    ),
    Rule(
        'extension.h',
        exclude_blacklisted(HasExtension('.h')), 
        FileAttribute.HEADER
    ),
    Rule(
        'extension.py',
        exclude_blacklisted(HasExtension('.py')), 
        FileAttribute.PYTHON
    ),
    Rule(
        'extension.c',
        exclude_blacklisted(HasExtension('.c')), 
        FileAttribute.C
    ),
    Rule(
        'extension.test.cc',
        exclude_blacklisted(HasExtension('.test.cc')), 
        FileAttribute.CPP_TEST
    ),
})
