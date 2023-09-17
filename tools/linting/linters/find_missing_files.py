#! /usr/bin/env python3

from lib.project import Project
from lib.library import Library
from lib.file import LogicalFile
from lib.file_type import FileType, file_type_of_path
from dataclasses import dataclass
from lib.path import AbsolutePath
from typing import Iterator, Dict, List, DefaultDict 
import json
from pathlib import Path
from lib.lint_response import LintResponse 
import sys
import logging
from lib.linter_helpers import add_verbosity_args, calculate_log_level

_l: logging.Logger

@dataclass(frozen=True)
class InvalidFileFound:
    file_type: FileType
    path: AbsolutePath

    @classmethod
    def create(cls, path: AbsolutePath):
        return cls(
            file_type=file_type_of_path(path),
            path=path,
        )

def find_invalid_files(project: Project) -> Iterator[InvalidFileFound]:
    for file in project.find_files_of_type(FileType.all()):
        lib = project.get_containing_library(file)
        if lib is None:
            yield InvalidFileFound.create(file)
        elif not lib.is_valid_path(file):
            yield InvalidFileFound.create(file)

@dataclass(frozen=True)
class MissingFile:
    file_type: FileType
    possible_paths: frozenset[AbsolutePath]
    because_of_paths: frozenset[AbsolutePath]

def find_missing_files(project: Project) -> Iterator[MissingFile]:
    for library in project.find_libraries():
        for logical_file in library.find_logical_files():
            for file_type in FileType.all():
                if not any(
                    p.is_file() for p in logical_file.paths_for_file_type(file_type)
                ):
                    because_of = frozenset({
                        p for p in logical_file.paths_for_file_types(file_type.others()) 
                        if p.is_file()
                    })

                    yield MissingFile(
                        file_type=file_type, 
                        possible_paths=logical_file.paths_for_file_type(file_type),
                        because_of_paths=because_of)

@dataclass
class Args:
    path: Path
    log_level: int

def run(args: Args) -> LintResponse:
    logging.basicConfig(level=args.log_level)

    global _l
    _l = logging.getLogger('find-missing-files')

    project = Project.for_path(args.path.absolute())

    invalid_files: Dict[FileType, List[str]] = { t: [] for t in FileType.all() }
    missing_files: Dict[FileType, List[Dict[str, List[AbsolutePath]]]] = DefaultDict(list)

    return_code = 0

    _l.info('Starting invalid search')
    for invalid_file in find_invalid_files(project):
        invalid_files[invalid_file.file_type].append(str(invalid_file.path))
        return_code = 1

    _l.info('Starting missing search')
    for missing_file in find_missing_files(project):
        missing_files[missing_file.file_type].append({
            'possible_paths': list(sorted(map(str, missing_file.possible_paths))),
            'because_of': list(sorted(map(str, missing_file.because_of_paths)))
        })
        return_code = 1
    _l.info('Done')

    if return_code != 0:
        return LintResponse.failure(json_data={
            'invalid_header_file_paths': invalid_files[FileType.HEADER],
            'invalid_source_file_paths': invalid_files[FileType.SOURCE],
            'invalid_test_file_paths': invalid_files[FileType.TEST],
            'missing_header_files': missing_files[FileType.HEADER],
            'missing_source_files': missing_files[FileType.SOURCE],
            'missing_test_files': missing_files[FileType.TEST],
        })
    else:
        return LintResponse.success()

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=Path, default=Path.cwd())
    add_verbosity_args(parser)
    args = parser.parse_args()

    response = run(Args(
        path=args.path,
        log_level=calculate_log_level(args),
    ))
    response.show()
    return response.return_code

if __name__ == '__main__':
    sys.exit(main())

