from ...layout.project import Project
from ...layout.cpp.preprocessor import with_unconventional_include_guards, rewrite_to_use_conventional_include_guard
from ...layout.cpp.file_group.file_group_component import FileGroupComponent 
from ..framework.response import Response
from pathlib import Path
import logging
from dataclasses import dataclass
from ..framework.helpers import check_unstaged_changes 
from ..framework.manager import Manager
from typing import FrozenSet, Set
from ..framework.specification import Specification, Method

_l = logging.getLogger(__name__)

def find_incorrect_include_guards(project: Project) -> FrozenSet[FileGroupComponent]:
    incorrect: Set[FileGroupComponent] = set()
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

def run(args: Args) -> Response:
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
        return Response.success(
            message='Fixed {len(incorrect)} header files'
        )
    else:
        if len(incorrect) > 0:
            return Response.failure(
                data=incorrect,
                json_data=list(sorted([str(inc.path) for inc in incorrect]))
            )

    return Response.success()

# def main() -> int:
#     import argparse

#     p = argparse.ArgumentParser()
#     p.add_argument('-p', '--path', type=Path, default=Path.cwd())
#     p.add_argument('--fix', action='store_true')
#     p.add_argument('--force', action='store_true')
#     p.add_argument('-v', '--verbose', action='count', default=0)
#     p.add_argument('-q', '--quiet', action='count', default=0)
#     add_verbosity_args(p)
#     args = p.parse_args()

#     result = run(Args(
#         path=args.path,
#         fix=args.fix,
#         force=args.force,
#         log_level=calculate_log_level(args)
#     ))

#     result.show()
#     return result.return_code

def register(mgr: Manager) -> None:
    mgr.register(Specification.create(
        name='include-guards',
        func=run,
        supported_methods={
            Method.FIX, Method.CHECK
        }
    ))
