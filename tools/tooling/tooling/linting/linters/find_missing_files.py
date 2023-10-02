from tooling.layout.project import Project
from tooling.layout.cpp.file_group.file_group import FileGroup
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from tooling.layout.file_type_inference.rules.rule import (
    Rule, HasAttribute, make_update_rules, OpaqueFunction, ExprExtra, Attrs, AncestorSatisfies
)
from tooling.layout.cpp.file_group.component_type import ComponentType 
from tooling.layout.path import AbsolutePath, with_all_suffixes_removed
from tooling.linting.framework.response import CheckResponse 
from tooling.linting.framework.specification import Specification 
from tooling.linting.framework.method import Method
from tooling.linting.framework.settings import Settings
from pathlib import Path

from enum import Enum, auto
from dataclasses import dataclass
from typing import Iterator, Dict, List, DefaultDict, FrozenSet, Tuple
import logging

_l = logging.getLogger(__name__)

is_supported_rule = Rule(
    HasAttribute(FileAttribute.CPP_FILE_GROUP_MEMBER),
    FileAttribute.IS_SUPPORTED_BY_FIND_MISSING_FILES_LINTER
)
header_update_rules = make_update_rules(
    is_supported=FileAttribute.IS_SUPPORTED_BY_FIND_MISSING_FILES_LINTER,
    old_incorrect=FileAttribute.ORIGINALLY_WAS_MISSING_HEADER_FILE,
    old_correct=FileAttribute.ORIGINALLY_HAD_HEADER_FILE,
    new_incorrect=FileAttribute.NOW_IS_MISSING_HEADER_FILE,
    new_correct=FileAttribute.NOW_HAS_HEADER_FILE
)
source_update_rules = make_update_rules(
    is_supported=FileAttribute.IS_SUPPORTED_BY_FIND_MISSING_FILES_LINTER,
    old_incorrect=FileAttribute.ORIGINALLY_WAS_MISSING_SOURCE_FILE,
    old_correct=FileAttribute.ORIGINALLY_HAD_SOURCE_FILE,
    new_incorrect=FileAttribute.NOW_IS_MISSING_SOURCE_FILE,
    new_correct=FileAttribute.NOW_HAS_SOURCE_FILE,
    did_fix=FileAttribute.DID_FIX_MISSING_SOURCE_FILE,
)
test_update_rules = make_update_rules(
    is_supported=FileAttribute.IS_SUPPORTED_BY_FIND_MISSING_FILES_LINTER,
    old_incorrect=FileAttribute.ORIGINALLY_WAS_MISSING_TEST_FILE,
    old_correct=FileAttribute.ORIGINALLY_HAD_TEST_FILE,
    new_incorrect=FileAttribute.NOW_IS_MISSING_TEST_FILE,
    new_correct=FileAttribute.NOW_HAS_TEST_FILE,
)
update_rules = header_update_rules.union(source_update_rules, test_update_rules)
common_rules = update_rules.union([is_supported_rule])

def get_check_missing_files_rules(project: Project) -> FrozenSet[Rule]:
    def _get_file_group_dirs(p: AbsolutePath, attrs: Attrs) -> Tuple[AbsolutePath, AbsolutePath, Path]:
        for parent in p.parents:
            if FileAttribute.CPP_FILE_GROUP_BASE in attrs(parent):
                break
        else:
            assert False

        file_group_base = parent
        assert FileAttribute.CPP_LIBRARY in attrs(file_group_base.parent)
        library_dir = file_group_base.parent
        include_dir = library_dir / 'include'
        src_dir = library_dir / 'src'
        file_group_path = with_all_suffixes_removed(p.relative_to(file_group_base))
        return include_dir, src_dir, file_group_path

    def _check_has_header(p: AbsolutePath, attrs: Attrs, extra: ExprExtra, project: Project = project) -> bool:
        include_dir, src_dir, file_group_path = _get_file_group_dirs(p, attrs)

        public_header_path = include_dir / file_group_path.with_suffix('.h')
        private_header_path = src_dir / file_group_path.with_suffix('.h')

        return public_header_path.is_file() or private_header_path.is_file()

    def _check_has_source(p: AbsolutePath, attrs: Attrs, extra: ExprExtra, project: Project = project) -> bool:
        include_dir, src_dir, file_group_path = _get_file_group_dirs(p, attrs)
        source_path = src_dir / file_group_path.with_suffix('.cc')
        return source_path.is_file()

    def _check_has_test(p: AbsolutePath, attrs: Attrs, extra: ExprExtra, project: Project = project) -> bool:
        include_dir, src_dir, file_group_path = _get_file_group_dirs(p, attrs)
        test_path = src_dir / file_group_path.with_suffix('.test.cc')
        return test_path.is_file()

    has_header_rule = Rule(
        OpaqueFunction(
            precondition=And.from_iter([
                HasAttribute(FileAttribute.IS_SUPPORTED_BY_FIND_MISSING_FILES_LINTER),
                AncestorSatisfies(FileAttribute.CPP_FILE_GROUP_BASE),
                AncestorSatisfies(FileATtribute.CPP_LIBRARY),
            ]),
            func=_check_has_header,
        ),
        FileAttribute.ORIGINALLY_HAD_HEADER_FILE,
    )

    has_source_rule = Rule(
        OpaqueFunction(
            precondition=And.from_iter([
                HasAttribute(FileAttribute.IS_SUPPORTED_BY_FIND_MISSING_FILES_LINTER),
                AncestorSatisfies(FileAttribute.CPP_FILE_GROUP_BASE),
                AncestorSatisfies(FileATtribute.CPP_LIBRARY),
            ]),
            func=_check_has_source,
        ),
        FileAttribute.ORIGINALLY_HAD_SOURCE_FILE,
    )

    has_test_rule = Rule(
        OpaqueFunction(
            precondition=And.from_iter([
                HasAttribute(FileAttribute.IS_SUPPORTED_BY_FIND_MISSING_FILES_LINTER),
                AncestorSatisfies(FileAttribute.CPP_FILE_GROUP_BASE),
                AncestorSatisfies(FileATtribute.CPP_LIBRARY),
            ]),
            func=_check_has_test,
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
        pass 

    fix_missing_src_file_rule = Rule(
        OpaqueFunction(
            precondition=HasAttribute(FileAttribute.ORIGINALLY_WAS_MISSING_SOURCE_FILE),
            func=_create_missing_source_file,
        ),
        FileAttribute.DID_FIX_MISSING_SOURCE_FILE
    )

    return get_check_missing_files_rules(project).union([fix_missing_src_file_rule])

def check_missing_files(project: Project) -> CheckResponse:
    pass

def fix_missing_files(project: Project) -> FixResponse:
    pass

@dataclass(frozen=True)
class InvalidFileFound:
    component_type: RequiredComponent
    path: AbsolutePath

    @classmethod
    def create(cls, path: AbsolutePath, attrs: FrozenSet[FileAttribute]) -> 'InvalidFileFound':
        return cls(
            component_type=RequiredComponent.from_file_attributes(attrs),
            path=path,
        )

def find_invalid_files(project: Project) -> Iterator[InvalidFileFound]:
    yield from (
        InvalidFileFound.create(path, project.file_types.for_path(path))
        for path in project.file_types.with_attr(FileAttribute.IS_INVALID_FILE)
    )

@dataclass(frozen=True)
class MissingFile:
    component_type: RequiredComponent
    possible_paths: FrozenSet[AbsolutePath]
    because_of_paths: FrozenSet[AbsolutePath]

def possible_paths_of_required_component(file_group: FileGroup, req: RequiredComponent) -> FrozenSet[AbsolutePath]:
    component_types = set()
    if req == RequiredComponent.HEADER:
        component_types.add(ComponentType.PUBLIC_HEADER)
        component_types.add(ComponentType.PRIVATE_HEADER)
    elif req == RequiredComponent.SOURCE:
        component_types.add(ComponentType.SOURCE)
    else:
        assert req == RequiredComponent.TEST
        component_types.add(ComponentType.TEST)

    return frozenset(
        file_group.path_of_component(ct) for ct in component_types
    )


def find_missing_files(project: Project) -> Iterator[MissingFile]:
    for library in project.cpp_code.libraries:
        for file_group in library.file_groups:
            for required_component in RequiredComponent.all():
                if not any(
                    p.is_file() for p in possible_paths_of_required_component(file_group, required_component)
                ):
                    yield MissingFile(
                        component_type=required_component, 
                        possible_paths=possible_paths_of_required_component(file_group, required_component),
                        because_of_paths=file_group.existing_components
                    )

def run(settings: Settings, project: Project, method: Method) -> CheckResponse:
    assert method == Method.CHECK
    invalid_files: Dict[RequiredComponent, List[str]] = { t: [] for t in RequiredComponent.all() }
    missing_files: Dict[RequiredComponent, List[Dict[str, List[str]]]] = DefaultDict(list)

    _l.info('Starting invalid search')
    for invalid_file in find_invalid_files(project):
        invalid_files[invalid_file.component_type].append(str(invalid_file.path))

    _l.info('Starting missing search')
    for missing_file in find_missing_files(project):
        missing_files[missing_file.component_type].append({
            'possible_paths': list(sorted(map(str, missing_file.possible_paths))),
            'because_of': list(sorted(map(str, missing_file.because_of_paths)))
        })
    _l.info('Done')

    num_invalid_files = sum(len(v) for v in invalid_files.values())
    num_missing_files = sum(len(v) for v in missing_files.values())

    return CheckResponse(
        num_errors=num_invalid_files + num_missing_files,
        json_data={
            'invalid_header_file_paths': invalid_files[RequiredComponent.HEADER],
            'invalid_source_file_paths': invalid_files[RequiredComponent.SOURCE],
            'invalid_test_file_paths': invalid_files[RequiredComponent.TEST],
            'missing_header_files': missing_files[RequiredComponent.HEADER],
            'missing_source_files': missing_files[RequiredComponent.SOURCE],
            'missing_test_files': missing_files[RequiredComponent.TEST],
        },
    )

spec = Specification.create(
    name='find_missing_files',
    func=run,
    supported_methods={ Method.CHECK }
)
