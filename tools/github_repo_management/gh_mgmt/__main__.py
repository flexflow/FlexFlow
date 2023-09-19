from .issues.cli import setup_issues_parser
from typing import Any
import logging
import argparse
import sys
from dataclasses import dataclass


def add_verbosity_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("-q", "--quiet", action="count", default=0)


def calculate_log_level(args: Any) -> int:
    LEVELS = [logging.DEBUG, logging.INFO, logging.WARN, logging.ERROR, logging.CRITICAL]
    default_verbosity = LEVELS.index(logging.WARN)
    verbosity = min(max(args.quiet - args.verbose + default_verbosity, 0), len(LEVELS))
    log_level = LEVELS[verbosity]
    return log_level


@dataclass(frozen=True)
class TopLevelArgs:
    json: bool


def main() -> int:
    PROGRAM_NAME = "repo"
    p = argparse.ArgumentParser(prog=PROGRAM_NAME)
    p.set_defaults(func=lambda *args, **kwargs: p.print_help())

    subparsers = p.add_subparsers()

    issues_p = subparsers.add_parser("issues")
    setup_issues_parser(issues_p)

    p.add_argument("--json", action="store_true")
    add_verbosity_args(p)
    args = p.parse_args()

    logging.basicConfig(level=calculate_log_level(args))

    _l = logging.getLogger(PROGRAM_NAME)

    return args.func()


if __name__ == "__main__":
    sys.exit(main())
