from ...layout.project import Project
from .response import Response
from typing import Optional, Any, List
import argparse
import logging

def check_unstaged_changes(project: Project, fix: bool, force: bool) -> Optional[Response]:
    if fix:
        if len(project.get_unstaged_changes()) > 0 and not force:
            return Response.failure(
                message=(
                    'Refusing to modify files because there are unstaged changes in git.\n'
                    'If you\'re really sure you trust this tool to not break your changes, '
                    'you can override this message with --force.'
                )
            )
    return None

def add_verbosity_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('-v', '--verbose', action='count', default=0)
    p.add_argument('-q', '--quiet', action='count', default=0)

def calculate_log_level(args: Any) -> int:
    LEVELS: List[int] = [
        logging.DEBUG,
        logging.INFO,
        logging.WARN,
        logging.ERROR,
        logging.CRITICAL
    ]
    default_verbosity = LEVELS.index(logging.WARN)
    verbosity = min(max(args.quiet - args.verbose + default_verbosity, 0), len(LEVELS))
    log_level = LEVELS[verbosity]
    return log_level
    
