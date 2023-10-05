from tooling.layout.project import Project
from tooling.layout.path import AbsolutePath
from tooling.linting.framework.response import CheckResponse, FixResponse, Response, ErrorResponse
from tooling.linting.framework.clang_tools import ClangToolsConfig, TOOL_CONFIGS, System, Arch, Tool, download_tool
from tooling.linting.framework.method import Method
from tooling.linting.framework.specification import Specification
from tooling.linting.framework.helpers import check_unstaged_changes 
from tooling.linting.framework.settings import Settings
from os import PathLike
from typing import Sequence, Union, FrozenSet
import logging
import subprocess
from tooling.layout.file_type_inference.file_attribute import FileAttribute

_l = logging.getLogger(__name__)

def find_compile_commands(project: Project) -> Union[AbsolutePath, ErrorResponse]:
    compile_command_paths = project.file_types.with_attr(FileAttribute.COMPILE_COMMANDS_JSON)

    if len(compile_command_paths) == 1:
        return list(compile_command_paths)[0]
    elif len(compile_command_paths) == 0:
        return ErrorResponse('Could not find any compile_commands.json')
    else:
        assert len(compile_command_paths) > 1
        return ErrorResponse(f'Found multiple compile_commands.json, disambiguation required: {list(sorted(compile_command_paths))}')

def run_clang_tidy(
    project: Project, 
    config: ClangToolsConfig, 
    compile_commands_path: AbsolutePath,
    args: Sequence[str], 
    files: Sequence[PathLike[str]]
) -> str:
    config_file = config.config_file_for_tool(Tool.clang_tidy)
    assert config_file is not None
    assert compile_commands_path.is_file()

    with config_file.open('r') as f:
        config_file_contents = f.read()

    command = [
        str(config.clang_tool_binary_path(Tool.clang_tidy)),
        '-p',
        str(compile_commands_path),
        f'--config={config_file_contents}',
        *args
    ]
    _l.debug(f'Running clang-tidy on {len(files)} files')
    return subprocess.check_output(command + [*files], stderr=subprocess.STDOUT).decode()


def get_files(project: Project) -> FrozenSet[AbsolutePath]:
    return project.file_types.with_all_of_attrs([FileAttribute.CPP, FileAttribute.IS_VALID_FILE])

def check_tidy(project: Project, config: ClangToolsConfig, compile_commands_path: AbsolutePath) -> Union[CheckResponse, ErrorResponse]:
    output = run_clang_tidy(
        project=project,
        config=config,
        args=[],
        compile_commands_path=compile_commands_path,
        files=list(get_files(project))
    )
    if len(output.splitlines()) > 0:
        return ErrorResponse(output)
    else:
        return CheckResponse(num_errors=0)

def fix_tidy(project: Project, config: ClangToolsConfig, compile_commands_path: AbsolutePath) -> Union[FixResponse, ErrorResponse]:
    output = run_clang_tidy(
        project=project, 
        config=config, 
        compile_commands_path=compile_commands_path, 
        args=['--fix'],
        files=list(get_files(project)),
    )
    if len(output.splitlines()) > 0:
        return ErrorResponse(output)
    else:
        return FixResponse(did_succeed=True)

def run(settings: Settings, project: Project, method: Method) -> Response:
    config = ClangToolsConfig(
        tools_dir=project.tools_download_dir, 
        tool_configs=TOOL_CONFIGS,
        system=System.get_current(),
        arch=Arch.get_current(),
    )

    is_fix = method == Method.FIX
    check_unstaged_changes(project, is_fix, settings.force)

    download_tool(Tool.clang_tidy, config)

    compile_commands_path = find_compile_commands(project)
    if isinstance(compile_commands_path, ErrorResponse):
        return compile_commands_path

    if is_fix:
        return fix_tidy(project, config, compile_commands_path)
    else:
        return check_tidy(project, config, compile_commands_path)


spec = Specification.create(
    name='clang_tidy',
    func=run,
    supported_methods=[Method.CHECK, Method.FIX]
)
