#! /usr/bin/env python3

import sys
from lib.project import Project
from lib.preprocessor import with_unconventional_include_guards, rewrite_to_use_conventional_include_guard
from lib.file import LogicalFileComponent
from lib.lint_response import LintResponse
from pathlib import Path
import json
import logging
from typing import Any, Optional
from dataclasses import dataclass
from lib.linter_helpers import check_unstaged_changes, add_verbosity_args, calculate_log_level 

_l: Any = None 

def find_incorrect_include_guards(project: Project) -> frozenset[LogicalFileComponent]:
    incorrect: Set[LogicalFileComponent] = set()
    for library in project.find_libraries():
        for logical_file in library.find_logical_files():
            incorrect |= with_unconventional_include_guards(logical_file)
    return frozenset(incorrect)

def fix_incorrect_include_guards(project: Project) -> None:
    for component in find_incorrect_include_guards(project):
        _l.info(f'Fixing include guard for component with path {component.path}')
        rewrite_to_use_conventional_include_guard(component)

@dataclass(frozen=True)
class Args:
    path: Path
    fix: bool
    force: bool
    log_level: int

def run(args: Args) -> LintResponse:
    logging.basicConfig(level=args.log_level)

    global _l
    _l = logging.getLogger('fix-include-guards')

    project = Project.for_path(args.path)

    error = check_unstaged_changes(project, args.fix, args.force)
    if error is not None:
        return error
 
    incorrect = find_incorrect_include_guards(project)

    if args.fix:
        fix_incorrect_include_guards(project)
        return LintResponse.success(
            message='Fixed {len(incorrect)} header files'
        )
    else:
        if len(incorrect) > 0:
            return LintResponse.failure(
                data=incorrect,
                json_data=list(sorted([str(inc.path) for inc in incorrect]))
            )

    return LintResponse.success()

def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('-p', '--path', type=Path, default=Path.cwd())
    p.add_argument('--fix', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('-v', '--verbose', action='count', default=0)
    p.add_argument('-q', '--quiet', action='count', default=0)
    add_verbosity_args(p)
    args = p.parse_args()

    result = run(Args(
        path=args.path,
        fix=args.fix,
        force=args.force,
        log_level=calculate_log_level(args)
    ))

    result.show()
    return result.return_code

if __name__ == '__main__':
    sys.exit(main())
