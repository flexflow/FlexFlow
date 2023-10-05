from tooling.layout.file_type_inference.rules.rule import Rule, And, IsDir, HasAttribute, DoesNotCreateNesting, IsFile, ChildSatisfies
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import FrozenSet

rules: FrozenSet[Rule] = frozenset({
    Rule(
        'build_directory.find',
        And.from_iter([
            IsDir(),
            ChildSatisfies('CMakeCache.txt', IsFile()),
            # DoesNotCreateNesting(HasAttribute(FileAttribute.BUILD_DIRECTORY))
        ]), 
        FileAttribute.BUILD_DIRECTORY
    )
})
