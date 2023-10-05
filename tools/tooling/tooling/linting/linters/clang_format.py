from tooling.layout.project import Project
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from tooling.layout.file_type_inference.rules.rule import Rule, OpaqueFunction, HasAnyOfAttributes, HasAttribute, HasAllOfAttributes, Attrs, ExprExtra, Not, And, make_update_rules
from tooling.linting.framework.response import CheckResponse, FixResponse, Response
from tooling.linting.framework.method import Method
from tooling.linting.framework.settings import Settings
from tooling.linting.framework.clang_tools import ClangToolsConfig, TOOL_CONFIGS, System, Arch, Tool, download_tool
from tooling.linting.framework.helpers import check_unstaged_changes 
from tooling.layout.path import AbsolutePath 
from tooling.linting.framework.specification import Specification
from os import PathLike

from typing import FrozenSet, Sequence
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
    if len(files) == 1:
        _l.debug(f'Running command {command} on 1 file: {files[0]}')
    else:
        _l.debug(f'Running command {command} on {len(files)} files')
    return subprocess.check_output(command + [*files], stderr=subprocess.STDOUT).decode()

supported_by_clang_format = Rule(
    'clang_format.is_supported',
    And.from_iter([
        HasAnyOfAttributes.from_iter([FileAttribute.CPP, FileAttribute.C, FileAttribute.HEADER]), 
        HasAttribute(FileAttribute.IS_VALID_FILE),
    ]),
    FileAttribute.SUPPORTED_BY_CLANG_FORMAT
)
    
update_rules = make_update_rules(
    'clang_format.update',
    is_supported=FileAttribute.SUPPORTED_BY_CLANG_FORMAT,
    old_incorrect=FileAttribute.WAS_IMPROPERLY_CLANG_FORMATTED,
    old_correct=FileAttribute.WAS_PROPERLY_CLANG_FORMATTED,
    new_incorrect=FileAttribute.IS_NOW_IMPROPERLY_CLANG_FORMATTED,
    new_correct=FileAttribute.IS_NOW_PROPERLY_CLANG_FORMATTED,
    did_fix=FileAttribute.DID_FIX_CLANG_FORMATTING,
)


def get_clang_check_rules(project: Project, config: ClangToolsConfig) -> FrozenSet[Rule]:
    def check_if_path_is_formatted(p: AbsolutePath, attrs: Attrs, extra: ExprExtra, project: Project = project, config: ClangToolsConfig = config) -> bool:
        output = run_clang_format(project, config, args=['--dry-run'], files=[p])
        extra.save(output)
        return len(output.splitlines()) == 0

    check_formatting_rule = Rule(
        'clang_format.check',
        OpaqueFunction(
            precondition=HasAttribute(FileAttribute.SUPPORTED_BY_CLANG_FORMAT), 
            func=check_if_path_is_formatted
        ),
        FileAttribute.WAS_PROPERLY_CLANG_FORMATTED
    )

    return update_rules.union([
        supported_by_clang_format,
        check_formatting_rule
    ])

def get_clang_fix_rules(project: Project, config: ClangToolsConfig) -> FrozenSet[Rule]:
    def format_path(p: AbsolutePath, attrs: Attrs, extra: ExprExtra, project: Project = project, config: ClangToolsConfig = config) -> bool:
        output = run_clang_format(project, config, args=['-i'], files=[p])
        extra.save(output)
        return True # clang format seems to always succeed

    run_formatter = Rule(
        'clang_format.fix',
        OpaqueFunction(
            precondition=HasAttribute(FileAttribute.SUPPORTED_BY_CLANG_FORMAT),
            func=format_path
        ),
        FileAttribute.DID_FIX_CLANG_FORMATTING
    )

    return get_clang_check_rules(project, config).union([
        supported_by_clang_format,
        run_formatter
    ])

def check_format(project: Project, config: ClangToolsConfig) -> CheckResponse:
    project.add_rules(get_clang_check_rules(project, config))
    failed_files = list(sorted(str(failed_file) for failed_file in project.file_types.with_attr(FileAttribute.IS_NOW_IMPROPERLY_CLANG_FORMATTED)))
    return CheckResponse(num_errors=len(failed_files), json_data=list(sorted(failed_files)))

def fix_format(project: Project, config: ClangToolsConfig) -> FixResponse:
    project.add_rules(get_clang_fix_rules(project, config))
    failed_files = list(sorted(str(failed_file) for failed_file in project.file_types.with_attr(FileAttribute.IS_NOW_IMPROPERLY_CLANG_FORMATTED)))
    fixed_files = list(sorted(str(fixed_file) for fixed_file in project.file_types.with_attr(FileAttribute.DID_FIX_CLANG_FORMATTING)))
    return FixResponse(
        did_succeed=len(failed_files) == 0,
        num_fixes=len(fixed_files),
        json_data={
            'failed': failed_files,
            'fixed': fixed_files
        }
    )

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
