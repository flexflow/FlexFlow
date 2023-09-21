from ...layout.project import Project
from ...layout.file_type import FileType
from ..framework.response import Response
from ..framework.clang_tools import ClangToolsConfig, TOOL_CONFIGS, System, Arch, Tool, download_tool
from ..framework.helpers import check_unstaged_changes 
from typing import FrozenSet, List
from pathlib import Path
from dataclasses import dataclass
import subprocess
import logging

_l = logging.getLogger(__name__)

def run_clang_format(project: Project, config: ClangToolsConfig, args: List[str], files: List[str]) -> str:
    style_file = project.root_path / config.config_file_for_tool(Tool.clang_format)
    assert style_file is not None
    command = [
        str(config.clang_tool_binary_path(Tool.clang_format)),
        f'--style=file:{style_file}',
        *args
    ]
    _l.debug(f'Running command {command}')
    return subprocess.check_output(command + ['-i', *files], stderr=subprocess.STDOUT).decode()

def get_files(project: Project) -> FrozenSet[Path]:
    languages = frozenset({FileType.CPU_CPP})
    return project.all_files(languages)

def check_format(project: Project, config: ClangToolsConfig) -> Response:
    failed_files: List[str] = []
    for file in get_files(project):
        output = run_clang_format(project, config, args=[
            '--dry-run'
        ], files=[str(file)])
        if len(output.splitlines()) > 0:
            failed_files.append(str(file))
    if len(failed_files) == 0:
        return Response.success()
    else:
        return Response.failure(
            json_data=list(sorted(failed_files))
        )

def fix_format(project: Project, config: ClangToolsConfig) -> LintResponse:
    assert False
    run_clang_format(config, [
        '-i', 
        *[str(path) for path in get_files(project)]
    ])

@dataclass(frozen=True)
class Args:
    fix: bool
    force: bool
    path: Path
    log_level: int


def run(args: Args) -> LintResponse:
    logging.basicConfig(level=args.log_level)
    global _l
    _l = logging.getLogger('format')
    project = Project.for_path(args.path)
    config = ClangToolsConfig(
        tools_dir=project.tools_download_dir,
        tool_configs=TOOL_CONFIGS,
        system=System.get_current(),
        arch=Arch.get_current()
    )

    check_unstaged_changes(project, args.fix, args.force)

    download_tool(Tool.clang_format, config)

    if args.fix:
        return fix_format(project, config)
    else:
        return check_format(project, config)

# def main() -> int:
#     import argparse

#     p = argparse.ArgumentParser()
#     p.add_argument('--path', type=Path, default=Path(__file__).parent)
#     p.add_argument('--fix', action='store_true')
#     p.add_argument('--force', action='store_true')
#     add_verbosity_args(p)
#     args = p.parse_args()

#     response = run(Args(
#         path=args.path,
#         fix=args.fix,
#         force=args.force,
#         log_level=calculate_log_level(args)
#     ))
    
#     response.show()
#     return response.return_code
