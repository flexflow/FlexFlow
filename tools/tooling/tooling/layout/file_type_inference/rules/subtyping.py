from tooling.layout.file_type_inference.rules.rule import Rule, HasAttribute, HasAnyOfAttributes
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import FrozenSet

rules: FrozenSet[Rule] = frozenset({
    Rule(HasAnyOfAttributes.from_iter([
        FileAttribute.CPP_PUBLIC_HEADER,
        FileAttribute.CPP_PRIVATE_HEADER,
        FileAttribute.CPP_FWDING_HEADER,
        FileAttribute.CPP_TEST,
        FileAttribute.IS_CPU_KERNEL,
        FileAttribute.IS_CUDA_KERNEL,
        FileAttribute.IS_HIP_KERNEL,
    ]), FileAttribute.CPP),
    Rule(HasAttribute(FileAttribute.INTERNAL_FFI_HEADER), FileAttribute.CPP_PRIVATE_HEADER),
    Rule(HasAttribute(FileAttribute.EXTERNAL_FFI_HEADER), FileAttribute.EXTERNAL_FFI_HEADER),
    Rule(HasAnyOfAttributes.from_iter([
        FileAttribute.CPP_SOURCE,
        FileAttribute.CPP_TEST
    ]), FileAttribute.IMPL),
    Rule(HasAttribute(FileAttribute.PYTHON_FF_TOOLS), FileAttribute.PYTHON),
})

