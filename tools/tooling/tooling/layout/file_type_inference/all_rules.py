from tooling.layout.file_type_inference.rules.infer import RuleCollection, InferenceResult
import tooling.layout.file_type_inference.rules.cmake as cmake
import tooling.layout.file_type_inference.rules.cpp_library as cpp_library
import tooling.layout.file_type_inference.rules.file_extension as file_extension
import tooling.layout.file_type_inference.rules.forwarding_header as forwarding_header
import tooling.layout.file_type_inference.rules.private_header as private_header
import tooling.layout.file_type_inference.rules.public_header as public_header
import tooling.layout.file_type_inference.rules.build_directory as build_directory
import tooling.layout.file_type_inference.rules.compile_commands as compile_commands
import tooling.layout.file_type_inference.rules.subtyping as subtyping
import tooling.layout.file_type_inference.rules.is_valid_file as is_valid_file
import tooling.layout.file_type_inference.rules.file_group as file_group
from tooling.layout.path import AbsolutePath

all_rules = RuleCollection(cmake.rules.union(
    cpp_library.rules,
    file_extension.rules,
    forwarding_header.rules,
    private_header.rules,
    public_header.rules,
    build_directory.rules,
    compile_commands.rules,
    subtyping.rules,
    is_valid_file.rules,
    file_group.rules
))

def run_all_rules(root: AbsolutePath) -> InferenceResult:
    return all_rules.run(root=root)
