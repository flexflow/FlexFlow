from tooling.layout.file_type_inference.rules.rule import Rule, IsDir, And, HasAttribute, ChildSatisfies, DoesNotCreateNesting, ParentSatisfies, IsNamed, AncestorSatisfies
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import FrozenSet

rules: FrozenSet[Rule] = frozenset({
    Rule(
        'cpp_library.find',
        And.from_iter([
            IsDir(),
            ChildSatisfies('src', IsDir()),
            ChildSatisfies('include', IsDir()),
            ChildSatisfies('CMakeLists.txt', HasAttribute(FileAttribute.CMAKELISTS)),
            # DoesNotCreateNesting(HasAttribute(FileAttribute.CPP_LIBRARY)),
        ]), 
        FileAttribute.CPP_LIBRARY
    ),
    Rule(
        'cpp_library.src',
        ParentSatisfies(HasAttribute(FileAttribute.CPP_LIBRARY)) & IsNamed('src'), 
        FileAttribute.CPP_LIBRARY_SRC_DIR
    ),
    Rule(
        'cpp_library.include',
        ParentSatisfies(HasAttribute(FileAttribute.CPP_LIBRARY)) & 
        IsNamed('include'), 
        FileAttribute.CPP_LIBRARY_INCLUDE_DIR
    ),
    Rule(
        'cpp_library.in_src',
        AncestorSatisfies(HasAttribute(FileAttribute.CPP_LIBRARY_SRC_DIR)), 
        FileAttribute.CPP_LIBRARY_IN_SRC
    ),
    Rule(
        'cpp_library.in_include',
        AncestorSatisfies(HasAttribute(FileAttribute.CPP_LIBRARY_INCLUDE_DIR)), 
        FileAttribute.CPP_LIBRARY_IN_INCLUDE
    ),
    Rule(
        'cpp_library.in_cpp_library',
        AncestorSatisfies(HasAttribute(FileAttribute.CPP_LIBRARY)), 
        FileAttribute.IN_CPP_LIBRARY
    ),
})
