from tooling.layout.file_type_inference.rules.rule import Rule, HasAttribute
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import FrozenSet

rules: FrozenSet[Rule] = frozenset({
    Rule(
        'public_header.find',
        HasAttribute(FileAttribute.HEADER) & HasAttribute(FileAttribute.CPP_LIBRARY_IN_INCLUDE),
        FileAttribute.CPP_PUBLIC_HEADER
    ),
})
