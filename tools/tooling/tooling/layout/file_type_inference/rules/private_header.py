from tooling.layout.file_type_inference.rules.rule import Rule, HasAttribute
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import FrozenSet

rules: FrozenSet[Rule] = frozenset({
    Rule(
        'private_header.find',
        HasAttribute(FileAttribute.HEADER) & HasAttribute(FileAttribute.CPP_LIBRARY_IN_SRC),
        FileAttribute.CPP_PRIVATE_HEADER
    ),
})
