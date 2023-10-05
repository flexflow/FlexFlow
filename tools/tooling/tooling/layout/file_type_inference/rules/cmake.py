from tooling.layout.file_type_inference.rules.rule import Rule, HasExtension, IsNamed
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import FrozenSet

rules: FrozenSet[Rule] = frozenset({
    Rule(
        'cmake.find_cmake_modules',
        HasExtension('.cmake'), FileAttribute.CMAKE
    ),
    Rule(
        'cmake.find_cmakelists',
        IsNamed('CMakeLists.txt'), 
        FileAttribute.CMAKELISTS
    )
})
