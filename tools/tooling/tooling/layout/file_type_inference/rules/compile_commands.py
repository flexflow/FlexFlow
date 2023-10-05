from tooling.layout.file_type_inference.rules.rule import Rule, IsNamed, IsFile
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import FrozenSet

rules: FrozenSet[Rule] = frozenset({
    Rule(
        'compile_commands.find',
        IsFile() & IsNamed('compile_commands.json'), 
        FileAttribute.COMPILE_COMMANDS_JSON
    )
})
