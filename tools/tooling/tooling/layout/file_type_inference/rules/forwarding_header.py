from tooling.layout.file_type_inference.rules.rule import Rule, And, HasAttribute, FileContentsSatisfy
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from tooling.layout.path import AbsolutePath
from typing import FrozenSet

def is_only_includes(path: AbsolutePath, contents: str) -> bool:
    lines = contents.splitlines()
    return all(line.startswith('#include') for line in lines)

rules: FrozenSet[Rule] = frozenset({
    Rule(
        'forwarding_header.find',
        And.from_iter([
            HasAttribute(FileAttribute.HEADER),
            FileContentsSatisfy(is_only_includes),
        ]), 
        FileAttribute.CPP_FWDING_HEADER
    ),
})
