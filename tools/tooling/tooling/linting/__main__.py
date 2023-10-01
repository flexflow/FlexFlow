import sys
from tooling.linting.linters.all_linters import all_linters
from tooling.cli.verbosity import add_verbosity_args, calculate_log_level
from tooling.linting.framework.response import FixResponse, CheckResponse, ErrorResponse
import argparse
from tooling.cli.parsing import instantiate, parser_root
from typing import Union, Mapping, Any
import logging
from tooling.layout.project import Project
from pathlib import Path
from tooling.linting.framework.settings import Settings

_l: logging.Logger

def all_linters_fix(raw_args: Any) -> Mapping[str, Union[FixResponse, ErrorResponse]]:
    logging.basicConfig(level=raw_args.log_level)
    _l = logging.getLogger(__name__)
    mgr = all_linters()
    _l.debug(f'Found project with root at {raw_args.project.root_path}')
    return mgr.fix(settings=raw_args.settings, project=raw_args.project)

def all_linters_check(raw_args: Any) -> Mapping[str, Union[CheckResponse, ErrorResponse]]:
    logging.basicConfig(level=raw_args.log_level)
    _l = logging.getLogger(__name__)
    mgr = all_linters()
    _l.debug(f'Found project with root at {raw_args.project.root_path}')
    return mgr.check(settings=raw_args.settings, project=raw_args.project)

def get_check_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return instantiate(p, func=all_linters_check)

def get_fix_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return instantiate(p, func=all_linters_fix)

def main() -> int:
    PROGRAM_NAME = "lint"

    _p = argparse.ArgumentParser(prog=PROGRAM_NAME, add_help=False)
    add_verbosity_args(_p)
    _p.add_argument('-f', '--force', action='store_true')

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
        print(responses)

    return 0

    

if __name__ == '__main__':
    sys.exit(main())
