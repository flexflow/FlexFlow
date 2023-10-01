from tooling.layout.project import Project
from tooling.layout.cpp.file_group.file_group import FileGroup
from tooling.layout.file_type import FileAttribute, FileAttributes 
from tooling.layout.cpp.file_group.component_type import ComponentType 
from tooling.layout.path import AbsolutePath
from tooling.linting.framework.response import CheckResponse 
from tooling.linting.framework.specification import Specification 
from tooling.linting.framework.method import Method
from tooling.linting.framework.settings import Settings

from enum import Enum, auto
from dataclasses import dataclass
from typing import Iterator, Dict, List, DefaultDict, FrozenSet
import logging

_l = logging.getLogger(__name__)

class RequiredComponent(Enum):
    HEADER = auto()
    SOURCE = auto()
    TEST = auto()

    @staticmethod
    def all() -> FrozenSet['RequiredComponent']:
        return frozenset({
            RequiredComponent.HEADER,
            RequiredComponent.SOURCE,
            RequiredComponent.TEST,
        })

    @staticmethod
    def from_file_attributes(attrs: FileAttributes) -> 'RequiredComponent':
        component: RequiredComponent 
        if attrs.implies(FileAttribute.CPP_SOURCE):
            component = RequiredComponent.SOURCE
        elif attrs.implies(FileAttribute.CPP_TEST):
            component = RequiredComponent.TEST
        else:
            assert attrs.implies_any_of([FileAttribute.CPP_PUBLIC_HEADER, FileAttribute.CPP_PRIVATE_HEADER])
            component = RequiredComponent.HEADER
        return component

@dataclass(frozen=True)
class InvalidFileFound:
    component_type: RequiredComponent
    path: AbsolutePath

    @classmethod
    def create(cls, path: AbsolutePath) -> 'InvalidFileFound':
        attrs = FileAttributes.for_path(path)

        return cls(
            component_type=RequiredComponent.from_file_attributes(attrs),
            path=path,
        )

def find_invalid_files(project: Project) -> Iterator[InvalidFileFound]:
    for file in project.files_satisfying(
        lambda p: FileAttributes.for_path(p).implies_any_of([
            FileAttribute.HEADER,
            FileAttribute.CPP_SOURCE,
            FileAttribute.CPP_TEST
        ])
    ):
        _l.debug(f'Checking if file {file} is invalid...')
        lib = project.cpp_code.get_containing_library(file)
        if lib is None:
            _l.debug('File was found to be invalid: no containing library')
            yield InvalidFileFound.create(file)
        elif not lib.is_valid_path(file):
            _l.debug('File was found to be invalid: invalid location in library')
            yield InvalidFileFound.create(file)
        else:
            _l.debug('File was found to be valid.')

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
    for library in project.cpp_code.find_libraries():
        for file_group in library.find_file_groups():
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
