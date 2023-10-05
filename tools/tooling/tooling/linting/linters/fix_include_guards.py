from tooling.layout.project import Project
from tooling.layout.cpp.preprocessor import (
    get_include_guard_var, set_include_guard_var
)
from tooling.layout.file_type_inference.rules.rule import (
    Rule, HasAttribute, HasAllOfAttributes, And, ExprExtra, OpaqueFunction, Attrs, HasAnyOfAttributes, Not, make_update_rules
)
from tooling.layout.file_type_inference.file_attribute import FileAttribute 
from tooling.layout.path import AbsolutePath
from tooling.linting.framework.response import Response, FixResponse, CheckResponse
from tooling.linting.framework.specification import Specification
from tooling.linting.framework.settings import Settings
from tooling.linting.framework.helpers import check_unstaged_changes 
from tooling.linting.framework.method import Method

import logging
from typing import FrozenSet

_l = logging.getLogger(__name__)

supported_by_fix_include_guards = Rule(
    'include_guards.is_supported',
    And.from_iter([
        HasAttribute(FileAttribute.IS_VALID_FILE),
        HasAttribute(FileAttribute.CPP_FILE_GROUP_MEMBER),
        HasAnyOfAttributes.from_iter([
            FileAttribute.CPP_PUBLIC_HEADER,
            FileAttribute.CPP_PRIVATE_HEADER
        ])
    ]),
    FileAttribute.SUPPORTED_BY_FIX_INCLUDE_GUARDS
)

update_rules = make_update_rules(
    'include_guards.update',
    is_supported=FileAttribute.SUPPORTED_BY_FIX_INCLUDE_GUARDS,
    old_incorrect=FileAttribute.ORIGINALLY_HAD_INCORRECT_INCLUDE_GUARD,
    old_correct=FileAttribute.ORIGINALLY_HAD_CORRECT_INCLUDE_GUARD,
    new_incorrect=FileAttribute.NOW_HAS_INCORRECT_INCLUDE_GUARD,
    new_correct=FileAttribute.NOW_HAS_CORRECT_INCLUDE_GUARD,
    did_fix=FileAttribute.DID_FIX_INCLUDE_GUARD
)

def get_include_guard_check_rules(project: Project) -> FrozenSet[Rule]:
    def _check_include_guards(p: AbsolutePath, attrs: Attrs, extra: ExprExtra, project: Project = project) -> bool:
        return get_include_guard_var(p) == project.include_guard_for_path(p)

    check_include_guards_rule = Rule(
        'include_guards.check',
        OpaqueFunction(
            precondition=HasAttribute(FileAttribute.SUPPORTED_BY_FIX_INCLUDE_GUARDS),
            func=_check_include_guards
        ),
        FileAttribute.ORIGINALLY_HAD_CORRECT_INCLUDE_GUARD
    )

    return update_rules.union([
        supported_by_fix_include_guards,
        check_include_guards_rule,
    ])

def get_include_guard_fix_rules(project: Project) -> FrozenSet[Rule]:
    def _fix_include_guards(p: AbsolutePath, attrs: Attrs, extra: ExprExtra, project: Project = project) -> bool:
        set_include_guard_var(p, project.include_guard_for_path(p))
        return True

    fix_include_guards_rule = Rule(
        'include_guards.fix',
        OpaqueFunction(
            precondition=HasAttribute(FileAttribute.ORIGINALLY_HAD_INCORRECT_INCLUDE_GUARD),
            func=_fix_include_guards
        ),
        FileAttribute.DID_FIX_INCLUDE_GUARD
    )

    return get_include_guard_check_rules(project).union({fix_include_guards_rule})

def check_include_guards(project: Project) -> CheckResponse:
    project.add_rules(get_include_guard_check_rules(project))
    incorrect_guards = [
        str(p) for p in project.file_types.with_attr(FileAttribute.NOW_HAS_INCORRECT_INCLUDE_GUARD)
    ]
    return CheckResponse(
        num_errors=len(incorrect_guards),
        json_data=list(sorted(incorrect_guards))
    )

def fix_include_guards(project: Project) -> FixResponse:
    project.add_rules(get_include_guard_fix_rules(project))

    still_incorrect_guards = [
        str(p) for p in project.file_types.with_attr(FileAttribute.NOW_HAS_INCORRECT_INCLUDE_GUARD)
    ]

    fixed_guards = [
        str(p) for p in project.file_types.with_attr(FileAttribute.DID_FIX_INCLUDE_GUARD)
    ]

    return FixResponse(
        did_succeed=(len(still_incorrect_guards) == 0),
        json_data={
            'failed_to_fix': sorted(list(still_incorrect_guards)),
            'fixed': sorted(list(fixed_guards))
        }
    )

def run(settings: Settings, project: Project, method: Method) -> Response:
    is_fix = method == Method.FIX
    error = check_unstaged_changes(project, is_fix, settings.force)
    if error is not None:
        return error
 
    if is_fix:
        return fix_include_guards(project)
    else:
        return check_include_guards(project)

spec = Specification.create(
    name='include_guards',
    func=run,
    supported_methods=[
        Method.FIX, Method.CHECK,
    ]
)
