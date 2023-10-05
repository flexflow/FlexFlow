from tooling.layout.file_type_inference.rules.rule import Rule, HasAttribute, HasAnyOfAttributes
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import FrozenSet

rules: FrozenSet[Rule] = frozenset({
    Rule(
        'subtyping.cpp',
        HasAnyOfAttributes.from_iter([
            FileAttribute.CPP_PUBLIC_HEADER,
            FileAttribute.CPP_PRIVATE_HEADER,
            FileAttribute.CPP_FWDING_HEADER,
            FileAttribute.CPP_TEST,
            FileAttribute.IS_CPU_KERNEL,
            FileAttribute.IS_CUDA_KERNEL,
            FileAttribute.IS_HIP_KERNEL,
        ]), 
        FileAttribute.CPP
    ),
    Rule(
        'subtyping.cpp_private_header',
        HasAttribute(FileAttribute.INTERNAL_FFI_HEADER), 
        FileAttribute.CPP_PRIVATE_HEADER
    ),
    Rule(
        'subtyping.external_ffi_header',
        HasAttribute(FileAttribute.EXTERNAL_FFI_HEADER), 
        FileAttribute.CPP_PUBLIC_HEADER
    ),
    Rule(
        'subtyping.cpp_impl',
        HasAnyOfAttributes.from_iter([
            FileAttribute.CPP_SOURCE,
            FileAttribute.CPP_TEST
        ]), 
        FileAttribute.IMPL
    ),
    Rule(
        'subtyping.python',
        HasAttribute(FileAttribute.PYTHON_FF_TOOLS), 
        FileAttribute.PYTHON
    ),
})

