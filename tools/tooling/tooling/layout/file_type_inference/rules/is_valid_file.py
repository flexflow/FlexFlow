from tooling.layout.file_type_inference.rules.rule import Rule, HasAttribute, Or, Not, HasAnyOfAttributes
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import FrozenSet

rules: FrozenSet[Rule] = frozenset({
    Rule(
        'invalid_file.find',
        Or.from_iter([
            HasAttribute(FileAttribute.IN_CPP_LIBRARY) & Not(HasAttribute(FileAttribute.CPP_FILE_GROUP_MEMBER)),
            HasAttribute(FileAttribute.CPP) & Not(HasAnyOfAttributes.from_iter([
                FileAttribute.CPP_LIBRARY_IN_SRC, FileAttribute.CPP_LIBRARY_IN_INCLUDE
            ])),
            HasAttribute(FileAttribute.CPP_LIBRARY_IN_SRC) & Not(HasAttribute(FileAttribute.CPP)),
            HasAttribute(FileAttribute.CPP_LIBRARY_IN_INCLUDE) & Not(HasAttribute(FileAttribute.CPP))
        ]), 
        FileAttribute.IS_INVALID_FILE
    ),
    Rule(
        'valid_file.from_invalid_files',
        Not(HasAttribute(FileAttribute.IS_INVALID_FILE)),
        FileAttribute.IS_VALID_FILE
    ),
})
