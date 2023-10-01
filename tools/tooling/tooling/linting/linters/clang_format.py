from tooling.layout.project import Project
from tooling.layout.file_type import FileAttributes, FileAttribute
from tooling.linting.framework.response import CheckResponse, FixResponse, Response
from tooling.linting.framework.method import Method
from tooling.linting.framework.settings import Settings
from tooling.linting.framework.clang_tools import ClangToolsConfig, TOOL_CONFIGS, System, Arch, Tool, download_tool
from tooling.linting.framework.helpers import check_unstaged_changes 
from tooling.layout.path import AbsolutePath 
from tooling.linting.framework.specification import Specification
from os import PathLike

from typing import FrozenSet, List, Sequence
import subprocess
import logging

_l = logging.getLogger(__name__)

def run_clang_format(project: Project, config: ClangToolsConfig, args: Sequence[str], files: Sequence[PathLike[str]]) -> str:
    config_file = config.config_file_for_tool(Tool.clang_format)
    assert config_file is not None
    style_file = project.root_path / config_file
    command = [
        str(config.clang_tool_binary_path(Tool.clang_format)),
        f'--style=file:{style_file}',
        *args
    ]
    _l.debug('Running command %s on %d files', command, len(files))
    return subprocess.check_output(command + [*files], stderr=subprocess.STDOUT).decode()

def get_files(project: Project) -> FrozenSet[AbsolutePath]:
    return project.files_satisfying(
        lambda p: FileAttributes.for_path(p).implies_any_of([FileAttribute.CPP, FileAttribute.C])
    )

def check_format(project: Project, config: ClangToolsConfig) -> CheckResponse:
    failed_files: List[str] = []
    for file in get_files(project):
        output = run_clang_format(project, config, args=[
            '--dry-run'
        ], files=[file])
        if len(output.splitlines()) > 0:
            failed_files.append(str(file))
    return CheckResponse(num_errors=len(failed_files), json_data=list(sorted(failed_files)))

def fix_format(project: Project, config: ClangToolsConfig) -> FixResponse:
    run_clang_format(project, config, args=['-i'], files=[path for path in get_files(project)])

    return FixResponse(did_succeed=True)

def run(settings: Settings, project: Project, method: Method) -> Response:
    config = ClangToolsConfig(
        tools_dir=project.tools_download_dir,
        tool_configs=TOOL_CONFIGS,
        system=System.get_current(),
        arch=Arch.get_current()
    )

    is_fix = method == Method.FIX
    check_unstaged_changes(project, is_fix, settings.force)

    download_tool(Tool.clang_format, config)

    if is_fix:
        return fix_format(project, config)
    else:
        return check_format(project, config)

spec = Specification.create(
    name='clang_format',
    func=run,
    supported_methods=[Method.CHECK, Method.FIX]
)
