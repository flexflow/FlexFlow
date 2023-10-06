from tooling.layout.project import Project
from tooling.layout.path import AbsolutePath
from tooling.layout.file_type_inference.rules.rule import Rule, HasAllOfAttributes, make_update_rules, OpaqueFunction, HasAttribute, Attrs, ExprExtra
from tooling.linting.framework.response import CheckResponse, FixResponse, Response, ErrorResponse
from tooling.linting.framework.clang_tools import ClangToolsConfig, TOOL_CONFIGS, System, Arch, Tool, download_tool
from tooling.linting.framework.method import Method
from tooling.linting.framework.specification import Specification
from tooling.linting.framework.helpers import check_unstaged_changes, jsonify_files_with_attr
from tooling.linting.framework.settings import Settings
from os import PathLike
from typing import Sequence, Union, FrozenSet, Any, Optional, Dict
import logging
import subprocess
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from pathlib import Path
import argparse
from tooling.cli.parsing import instantiate
from tooling.linting.linters.fix_include_guards import get_include_guard_check_rules, get_include_guard_fix_rules 

_l: logging.Logger = logging.getLogger(__name__)

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
    config_file = project.root_path / config.config_file_for_tool(Tool.clang_tidy)
    _l.debug(f'clang-tidy config should be located at {config_file}')
    assert config_file is not None
    assert compile_commands_path.is_file()

    with config_file.open('r') as f:
        config_file_contents = f.read()

    try:
        command = [
            str(config.clang_tool_binary_path(Tool.clang_tidy)),
            '-p',
            str(compile_commands_path),
            f'--config={config_file_contents}',
            *args
        ]
        _l.debug(f'Running clang-tidy on {len(files)} files')
        return subprocess.check_output(command + [*files], stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        _l.exception(f'clang-tidy failed with output: {e.stdout}')
        raise

supported_by_clang_tidy = Rule(
    'clang_tidy.is_supported',
    HasAllOfAttributes.from_iter([
        FileAttribute.CPP, 
        FileAttribute.IS_VALID_FILE,
        FileAttribute.NOW_HAS_CORRECT_INCLUDE_GUARD,
    ]),
    FileAttribute.IS_SUPPORTED_BY_CLANG_TIDY,
)

update_rules = make_update_rules(
    'clang_tidy.update',
    is_supported=FileAttribute.IS_SUPPORTED_BY_CLANG_TIDY,
    old_incorrect=FileAttribute.ORIGINALLY_FAILED_CLANG_TIDY_CHECKS,
    old_correct=FileAttribute.ORIGINALLY_PASSED_CLANG_TIDY_CHECKS,
    new_incorrect=FileAttribute.NOW_FAILS_CLANG_TIDY_CHECKS,
    new_correct=FileAttribute.NOW_PASSES_CLANG_TIDY_CHECKS,
    did_fix=FileAttribute.DID_FIX_CLANG_TIDY_FAILED_CHECKS,
)

def get_clang_tidy_check_rules(project: Project, config: ClangToolsConfig, compile_commands_path: AbsolutePath) -> FrozenSet[Rule]:
    def passes_clang_tidy_check(
        p: AbsolutePath, 
        attrs: Attrs, 
        extra: ExprExtra,
        project: Project = project,
        config: ClangToolsConfig = config,
        compile_commands_path: AbsolutePath = compile_commands_path,
    ) -> bool:
        output = run_clang_tidy(
            project=project,
            config=config,
            args=[],
            compile_commands_path=compile_commands_path,
            files=[p],
        )
        extra.save(output)
        return len(output.splitlines()) == 0

    check_clang_tidy_rule = Rule(
        'clang_tidy.check',
        OpaqueFunction(
            precondition=HasAttribute(FileAttribute.IS_SUPPORTED_BY_CLANG_TIDY),
            func=passes_clang_tidy_check,
        ),
        FileAttribute.ORIGINALLY_PASSED_CLANG_TIDY_CHECKS,
    )

    return update_rules.union([
        supported_by_clang_tidy,
        check_clang_tidy_rule,
        *get_include_guard_check_rules(project),
    ])

def get_clang_tidy_fix_rules(project: Project, config: ClangToolsConfig, compile_commands_path: AbsolutePath) -> FrozenSet[Rule]:
    def run_clang_tidy_in_fix_mode(
        p: AbsolutePath,
        attrs: Attrs,
        extra: ExprExtra,
        project: Project = project,
        config: ClangToolsConfig = config,
        compile_commands_path: AbsolutePath = compile_commands_path
    ) -> bool:
        output = run_clang_tidy(
            project=project, 
            config=config, 
            compile_commands_path=compile_commands_path, 
            args=['--fix'],
            files=[p],
        )
        extra.save(output)
        return True

    clang_tidy_fix_rule = Rule(
        'clang_tidy_fix_rule',
        OpaqueFunction(
            precondition=HasAttribute(FileAttribute.IS_SUPPORTED_BY_CLANG_TIDY),
            func=run_clang_tidy_in_fix_mode,
        ),
        FileAttribute.DID_FIX_CLANG_TIDY_FAILED_CHECKS,
    )

    return update_rules.union([
        supported_by_clang_tidy,
        clang_tidy_fix_rule,
        *get_include_guard_fix_rules(project),
    ])

def check_tidy(project: Project, config: ClangToolsConfig, compile_commands_path: AbsolutePath) -> CheckResponse:
    project.add_rules(get_clang_tidy_check_rules(project, config, compile_commands_path))
    failed_files = jsonify_files_with_attr(project, FileAttribute.NOW_FAILS_CLANG_TIDY_CHECKS)
    return CheckResponse(num_errors=len(failed_files), json_data=failed_files)

def fix_tidy(project: Project, config: ClangToolsConfig, compile_commands_path: AbsolutePath) -> FixResponse:
    project.add_rules(get_clang_tidy_fix_rules(project, config, compile_commands_path))
    failed_files = jsonify_files_with_attr(project, FileAttribute.NOW_FAILS_CLANG_TIDY_CHECKS)
    fixed_files = jsonify_files_with_attr(project, FileAttribute.DID_FIX_CLANG_TIDY_FAILED_CHECKS)
    return FixResponse(
        did_succeed=len(failed_files) == 0,
        num_fixes=len(fixed_files),
        json_data={
            'failed': failed_files,
            'fixed': fixed_files,
        },
    )

def run(settings: Settings, project: Project, method: Method, compile_commands_path: Optional[AbsolutePath] = None) -> Response:
    config = ClangToolsConfig(
        tools_dir=project.tools_download_dir, 
        tool_configs=TOOL_CONFIGS,
        system=System.get_current(),
        arch=Arch.get_current(),
    )

    is_fix = method == Method.FIX
    check_unstaged_changes(project, is_fix, settings.force)

    download_tool(Tool.clang_tidy, config)

    if compile_commands_path is None:
        find_result = find_compile_commands(project)
        if isinstance(find_result, ErrorResponse):
            return find_result
        else:
            compile_commands_path = find_result

    if is_fix:
        return fix_tidy(project, config, compile_commands_path)
    else:
        return check_tidy(project, config, compile_commands_path)

def main_tidy(raw_args: Any) -> Dict[str, Response]:
    _l = raw_args.setup_logging(__name__, raw_args)
    return {
        raw_args.linter_name : run(settings=raw_args.settings, project=raw_args.project, method=raw_args.method, compile_commands_path=raw_args.compile_commands)
    }

def get_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument('--compile-commands', type=lambda s: AbsolutePath.create(Path(s), Path.cwd()), default=None)

    return instantiate(
        p,
        func=main_tidy,
    )

spec = Specification.create(
    name='clang_tidy',
    func=run,
    supported_methods=[Method.CHECK, Method.FIX]
)
