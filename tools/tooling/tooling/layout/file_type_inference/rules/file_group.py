from tooling.layout.file_type_inference.rules.rule import Rule, HasAnyOfAttributes, AncestorSatisfies, HasAttribute
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import FrozenSet

rules: FrozenSet[Rule] = frozenset({
    Rule(
        'file_group_base.find',
        HasAnyOfAttributes.from_iter([FileAttribute.CPP_LIBRARY_INCLUDE_DIR, FileAttribute.CPP_LIBRARY_SRC_DIR]),
        FileAttribute.CPP_FILE_GROUP_BASE
    ),
    Rule(
        'file_group_member.find',
        HasAnyOfAttributes.from_iter([
            FileAttribute.CPP_SOURCE, FileAttribute.CPP_TEST, FileAttribute.CPP_PRIVATE_HEADER, FileAttribute.CPP_PUBLIC_HEADER
        ]) & AncestorSatisfies(HasAttribute(FileAttribute.CPP_FILE_GROUP_BASE)),
        FileAttribute.CPP_FILE_GROUP_MEMBER
    ),
})
