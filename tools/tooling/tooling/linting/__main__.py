import sys
from tooling.linting.linters.all_linters import all_linters, SpecificLinter
from tooling.cli.verbosity import add_verbosity_args, calculate_log_level
from tooling.linting.framework.response import FixResponse, CheckResponse, ErrorResponse
import argparse
from tooling.cli.parsing import instantiate, parser_root
from typing import Union, Mapping, Any, Callable
import logging
from tooling.layout.project import Project
from pathlib import Path
from tooling.linting.framework.settings import Settings
from enum import Enum 
import json

_l: logging.Logger

def setup_logging(raw_args: Any) -> logging.Logger:
    logging.basicConfig(level=raw_args.log_level)
    return logging.getLogger(__name__)

def all_linters_fix(raw_args: Any) -> Mapping[str, Union[FixResponse, ErrorResponse]]:
    _l = setup_logging(raw_args)
    mgr = all_linters()
    _l.debug(f'Found project with root at {raw_args.project.root_path}')
    return mgr.fix(settings=raw_args.settings, project=raw_args.project)

def all_linters_check(raw_args: Any) -> Mapping[str, Union[CheckResponse, ErrorResponse]]:
    _l = setup_logging(raw_args)
    mgr = all_linters()
    _l.debug(f'Found project with root at {raw_args.project.root_path}')
    return mgr.check(settings=raw_args.settings, project=raw_args.project)

def specific_linter_fix(linter: SpecificLinter) -> Callable[[Any], Mapping[str, Union[FixResponse, ErrorResponse]]]:
    def _f(raw_args: Any, linter: SpecificLinter=linter) -> Mapping[str, Union[FixResponse, ErrorResponse]]:
        _l = setup_logging(raw_args)
        mgr = linter.get_manager()
        _l.debug(f'Found project with root at {raw_args.project.root_path}')
        return mgr.fix(settings=raw_args.settings, project=raw_args.project)
    return _f

def specific_linter_check(linter: SpecificLinter) -> Callable[[Any], Mapping[str, Union[CheckResponse, ErrorResponse]]]:
    def _f(raw_args: Any, linter: SpecificLinter=linter) -> Mapping[str, Union[CheckResponse, ErrorResponse]]:
        _l = setup_logging(raw_args)
        mgr = linter.get_manager()
        _l.debug(f'Found project with root at {raw_args.project.root_path}')
        return mgr.check(settings=raw_args.settings, project=raw_args.project)
    return _f

def specific_linter_check_parser(linter: SpecificLinter) -> Callable[[argparse.ArgumentParser], argparse.ArgumentParser]:
    def _f(p: argparse.ArgumentParser, linter: SpecificLinter = linter) -> argparse.ArgumentParser:
        return instantiate(p, func=specific_linter_check(linter))
    return _f

def specific_linter_fix_parser(linter: SpecificLinter) -> Callable[[argparse.ArgumentParser], argparse.ArgumentParser]:
    def _f(p: argparse.ArgumentParser, linter: SpecificLinter = linter) -> argparse.ArgumentParser:
        return instantiate(p, func=specific_linter_fix(linter))
    return _f

def get_check_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    specific_linter_parsers = {
        linter.name : specific_linter_check_parser(linter)
        for linter in SpecificLinter
        if linter.supports_check
    }
    return instantiate(p, func=all_linters_check, **specific_linter_parsers)

def get_fix_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    specific_linter_parsers = {
        linter.name : specific_linter_fix_parser(linter)
        for linter in SpecificLinter
        if linter.supports_fix
    }
    return instantiate(p, func=all_linters_fix, **specific_linter_parsers)

class OutputFormat(Enum):
    plaintext = 'plaintext'
    json = 'json'

    def __str__(self) -> str:
        return self.value

def main() -> int:
    PROGRAM_NAME = "lint"

    _p = argparse.ArgumentParser(prog=PROGRAM_NAME, add_help=False)
    add_verbosity_args(_p)
    _p.add_argument('--force', action='store_true')
    _p.add_argument('-f', '--format', type=OutputFormat, default=OutputFormat.plaintext, choices=list(OutputFormat))

    p = parser_root(instantiate(
        _p,
        fix=get_fix_parser,
        check=get_check_parser,
    ))

    args = p.parse_args()
    args.log_level = calculate_log_level(args)
    args.project = Project.for_path(Path(__file__))
    args.settings = Settings(force=args.force)

    responses = args.func(args)

    if not args.silent:
        if args.format == OutputFormat.plaintext:
            for k, r in responses.items():
                if isinstance(r, CheckResponse):
                    print(f'- {k} found {r.num_errors} errors')
                elif isinstance(r, FixResponse):
                    print(f'- {k} fixed {r.num_fixes} errors')
                elif isinstance(r, ErrorResponse):
                    print(f'- {k} crashed with the following error:')
                    print('  ' + r.message)
        elif args.format == OutputFormat.json:
            json_output = {}
            for k, r in responses.items():
                json_output[k] = {
                    '_type': r.__class__.__name__,
                    'data': r.as_json()
                }
            print(json.dumps(json_output, indent=2))

    return 0

    

if __name__ == '__main__':
    sys.exit(main())
