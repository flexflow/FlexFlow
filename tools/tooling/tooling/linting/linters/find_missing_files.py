from tooling.layout.project import Project
from tooling.layout.cpp.file_group.file_group import FileGroup
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from tooling.layout.file_type_inference.rules.rule import (
    Rule, HasAttribute, make_update_rules, OpaqueFunction, ExprExtra, Attrs, AncestorSatisfies, And
)
from tooling.layout.path import AbsolutePath, with_all_suffixes_removed
from tooling.linting.framework.response import CheckResponse, FixResponse, Response
from tooling.linting.framework.specification import Specification 
from tooling.linting.framework.method import Method
from tooling.linting.framework.settings import Settings
from tooling.linting.framework.helpers import check_unstaged_changes
from pathlib import Path
from tooling.json import Json

from typing import Dict, List, DefaultDict, FrozenSet, Tuple
import logging

_l = logging.getLogger(__name__)

is_supported_rule = Rule(
    'missing_files.is_supported',
    HasAttribute(FileAttribute.CPP_FILE_GROUP_MEMBER),
    FileAttribute.IS_SUPPORTED_BY_FIND_MISSING_FILES_LINTER
)
header_update_rules = make_update_rules(
    'missing_files.header.update',
    is_supported=FileAttribute.IS_SUPPORTED_BY_FIND_MISSING_FILES_LINTER,
    old_incorrect=FileAttribute.ORIGINALLY_WAS_MISSING_HEADER_FILE,
    old_correct=FileAttribute.ORIGINALLY_HAD_HEADER_FILE,
    new_incorrect=FileAttribute.NOW_IS_MISSING_HEADER_FILE,
    new_correct=FileAttribute.NOW_HAS_HEADER_FILE
)
source_update_rules = make_update_rules(
    'missing_files.source.update',
    is_supported=FileAttribute.IS_SUPPORTED_BY_FIND_MISSING_FILES_LINTER,
    old_incorrect=FileAttribute.ORIGINALLY_WAS_MISSING_SOURCE_FILE,
    old_correct=FileAttribute.ORIGINALLY_HAD_SOURCE_FILE,
    new_incorrect=FileAttribute.NOW_IS_MISSING_SOURCE_FILE,
    new_correct=FileAttribute.NOW_HAS_SOURCE_FILE,
    did_fix=FileAttribute.DID_FIX_MISSING_SOURCE_FILE,
)
test_update_rules = make_update_rules(
    'missing_files.test.update',
    is_supported=FileAttribute.IS_SUPPORTED_BY_FIND_MISSING_FILES_LINTER,
    old_incorrect=FileAttribute.ORIGINALLY_WAS_MISSING_TEST_FILE,
    old_correct=FileAttribute.ORIGINALLY_HAD_TEST_FILE,
    new_incorrect=FileAttribute.NOW_IS_MISSING_TEST_FILE,
    new_correct=FileAttribute.NOW_HAS_TEST_FILE,
)
update_rules = header_update_rules.union(source_update_rules, test_update_rules)
common_rules = update_rules.union([is_supported_rule])

def _get_file_group_dirs(p: AbsolutePath, attrs: Attrs) -> Tuple[AbsolutePath, AbsolutePath, AbsolutePath, Path]:
    for parent in p.parents:
        if FileAttribute.CPP_FILE_GROUP_BASE in attrs(parent):
            break
    else:
        _l.error(f'Could not find file group base of {p}')
        assert False

    file_group_base = parent
    _l.debug(f'Found file group base {file_group_base} for path {p}')
    assert FileAttribute.CPP_LIBRARY in attrs(file_group_base.parent)
    library_dir = file_group_base.parent
    include_dir = library_dir / 'include'
    src_dir = library_dir / 'src'
    file_group_path = with_all_suffixes_removed(p.relative_to(file_group_base))
    return library_dir, include_dir, src_dir, file_group_path

_get_file_group_dirs_deps = frozenset([
    FileAttribute.CPP_FILE_GROUP_BASE,
    FileAttribute.CPP_LIBRARY,
])


def get_check_missing_files_rules(project: Project) -> FrozenSet[Rule]:
    def _check_has_header(p: AbsolutePath, attrs: Attrs, extra: ExprExtra, project: Project = project) -> bool:
        _, include_dir, src_dir, file_group_path = _get_file_group_dirs(p, attrs)

        public_header_path = include_dir / file_group_path.with_suffix('.h')
        private_header_path = src_dir / file_group_path.with_suffix('.h')

        return public_header_path.is_file() or private_header_path.is_file()

    def _check_has_source(p: AbsolutePath, attrs: Attrs, extra: ExprExtra, project: Project = project) -> bool:
        _, include_dir, src_dir, file_group_path = _get_file_group_dirs(p, attrs)
        source_path = src_dir / file_group_path.with_suffix('.cc')
        return source_path.is_file()

    def _check_has_test(p: AbsolutePath, attrs: Attrs, extra: ExprExtra, project: Project = project) -> bool:
        _, include_dir, src_dir, file_group_path = _get_file_group_dirs(p, attrs)
        test_path = src_dir / file_group_path.with_suffix('.test.cc')
        return test_path.is_file()

    has_header_rule = Rule(
        'missing_files.has_header',
        OpaqueFunction(
            precondition=And.from_iter([
                HasAttribute(FileAttribute.IS_SUPPORTED_BY_FIND_MISSING_FILES_LINTER),
                AncestorSatisfies(HasAttribute(FileAttribute.CPP_FILE_GROUP_BASE)),
                AncestorSatisfies(HasAttribute(FileAttribute.CPP_LIBRARY)),
            ]),
            func=_check_has_header,
            extra_inputs=_get_file_group_dirs_deps,

        ),
        FileAttribute.ORIGINALLY_HAD_HEADER_FILE,
    )

    has_source_rule = Rule(
        'missing_files.has_source', 
        OpaqueFunction(
            precondition=And.from_iter([
                HasAttribute(FileAttribute.IS_SUPPORTED_BY_FIND_MISSING_FILES_LINTER),
                AncestorSatisfies(HasAttribute(FileAttribute.CPP_FILE_GROUP_BASE)),
                AncestorSatisfies(HasAttribute(FileAttribute.CPP_LIBRARY)),
            ]),
            func=_check_has_source,
            extra_inputs=_get_file_group_dirs_deps,
        ),
        FileAttribute.ORIGINALLY_HAD_SOURCE_FILE,
    )

    has_test_rule = Rule(
        'missing_files.has_test',
        OpaqueFunction(
            precondition=And.from_iter([
                HasAttribute(FileAttribute.IS_SUPPORTED_BY_FIND_MISSING_FILES_LINTER),
                AncestorSatisfies(HasAttribute(FileAttribute.CPP_FILE_GROUP_BASE)),
                AncestorSatisfies(HasAttribute(FileAttribute.CPP_LIBRARY)),
            ]),
            func=_check_has_test,
            extra_inputs=_get_file_group_dirs_deps,
        ),
        FileAttribute.ORIGINALLY_HAD_TEST_FILE,
    )

    return common_rules.union([
        has_header_rule,
        has_source_rule,
        has_test_rule,
    ])
    

def get_fix_missing_files_rules(project: Project) -> FrozenSet[Rule]:
    def _create_missing_source_file(p: AbsolutePath, attrs: Attrs, extra: ExprExtra, project: Project = project) -> bool:
        library_dir, include_dir, src_dir, file_group_path = _get_file_group_dirs(p, attrs)
        library = project.cpp_code.library_for_path(library_dir)
        assert library is not None
        file_group = FileGroup(library, file_group_path)
        if not file_group.source_file.exists():
            file_group.source_path.parent.mkdir(exist_ok=True, parents=True)
            with file_group.source_path.open('w') as f:
                if file_group.public_header.exists():
                    f.write('#include "{file_group.public_header_include_path}"')
                if file_group.private_header.exists():
                    f.write('#include "{file_group.private_header_include_path}"')
        return True

    fix_missing_src_file_rule = Rule(
        'missing_files.create_missing_source',
        OpaqueFunction(
            precondition=And.from_iter([
                HasAttribute(FileAttribute.ORIGINALLY_WAS_MISSING_SOURCE_FILE),
                HasAttribute(FileAttribute.ORIGINALLY_HAD_HEADER_FILE),
            ]),
            func=_create_missing_source_file,
            extra_inputs=_get_file_group_dirs_deps,
        ),
        FileAttribute.DID_FIX_MISSING_SOURCE_FILE
    )

    return get_check_missing_files_rules(project).union([fix_missing_src_file_rule])

def _get_json_data(project: Project) -> Tuple[int, Json]:
    missing_header_files = list(sorted([
        str(p) for p in project.file_types.with_attr(FileAttribute.NOW_IS_MISSING_HEADER_FILE)
    ]))
    missing_test_files = list(sorted([
        str(p) for p in project.file_types.with_attr(FileAttribute.NOW_IS_MISSING_TEST_FILE)
    ]))
    missing_source_files = list(sorted([
        str(p) for p in project.file_types.with_attr(FileAttribute.NOW_IS_MISSING_SOURCE_FILE)
    ]))
    num_missing = len(set(missing_header_files).union(missing_test_files, missing_source_files))
    return (
        num_missing,
        {
            'missing_header_files': missing_header_files,
            'missing_test_files': missing_test_files,
            'missing_source_files': missing_source_files,
        }
    )

def check_missing_files(project: Project) -> CheckResponse:
    project.add_rules(get_check_missing_files_rules(project))
    num_errors, json_data = _get_json_data(project)
    return CheckResponse(
        num_errors=num_errors,
        json_data=json_data
    )

def fix_missing_files(project: Project) -> FixResponse:
    project.add_rules(get_fix_missing_files_rules(project))
    num_errors, json_data = _get_json_data(project)
    fixed = list(sorted([
        str(p) for p in project.file_types.with_attr(FileAttribute.DID_FIX_MISSING_SOURCE_FILE)
    ]))
    return FixResponse(
        did_succeed=(num_errors == 0),
        num_fixes=len(fixed),
        json_data=json_data
    )

def run(settings: Settings, project: Project, method: Method) -> Response:
    is_fix = (method == Method.FIX)
    error = check_unstaged_changes(project, is_fix, settings.force)
    if error is not None:
        return error

    if is_fix:
        return fix_missing_files(project)
    else:
        return check_missing_files(project)

spec = Specification.create(
    name='find_missing_files',
    func=run,
    supported_methods={ Method.CHECK, Method.FIX }
)
