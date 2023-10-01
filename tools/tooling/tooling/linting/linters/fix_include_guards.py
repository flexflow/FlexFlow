from tooling.layout.project import Project
from tooling.layout.cpp.preprocessor import (
    with_unconventional_include_guards, rewrite_to_use_conventional_include_guard
)
from tooling.layout.cpp.file_group.file_group_component import FileGroupComponent 
from tooling.linting.framework.response import Response, FixResponse, CheckResponse
from tooling.linting.framework.specification import Specification
from tooling.linting.framework.settings import Settings
from tooling.linting.framework.helpers import check_unstaged_changes 
from tooling.linting.framework.method import Method

import logging
from typing import FrozenSet, Set

_l = logging.getLogger(__name__)

def find_incorrect_include_guards(project: Project) -> FrozenSet[FileGroupComponent]:
    incorrect: Set[FileGroupComponent] = set()
    for library in project.cpp_code.find_libraries():
        for logical_file in library.find_file_groups():
            incorrect |= with_unconventional_include_guards(logical_file)
    return frozenset(incorrect)

def fix_incorrect_include_guards(project: Project) -> None:
    for component in find_incorrect_include_guards(project):
        _l.info(f'Fixing include guard for component with path {component.path}')
        rewrite_to_use_conventional_include_guard(component)

def run(settings: Settings, project: Project, method: Method) -> Response:
    is_fix = method == Method.FIX
    error = check_unstaged_changes(project, is_fix, settings.force)
    if error is not None:
        return error
 
    incorrect = find_incorrect_include_guards(project)

    incorrect_jsonable = list(sorted([str(inc.path) for inc in incorrect]))
    if is_fix:
        fix_incorrect_include_guards(project)
        return FixResponse(
            did_succeed=True,
            num_fixes=len(incorrect),
            json_data=incorrect_jsonable,
        )
    else:
        return CheckResponse(
            num_errors=len(incorrect),
            json_data=incorrect_jsonable,
        )

spec = Specification.create(
    name='include_guards',
    func=run,
    supported_methods=[
        Method.FIX, Method.CHECK,
    ]
)
