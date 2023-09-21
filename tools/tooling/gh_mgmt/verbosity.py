import argparse
import logging
from typing import Any

def add_verbosity_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("-q", "--quiet", action="count", default=0)
    p.add_argument("--silent", action="store_true")

def calculate_log_level(args: Any) -> int:
    LEVELS = [logging.DEBUG, logging.INFO, logging.WARN, logging.ERROR, logging.CRITICAL]
    default_verbosity = LEVELS.index(logging.WARN)
    verbosity = min(max(args.quiet - args.verbose + default_verbosity, 0), len(LEVELS))
    log_level = LEVELS[verbosity]
    if args.silent:
        log_level = max(LEVELS) + 1
    return log_level

