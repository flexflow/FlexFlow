import argparse
from .triage.cli import setup_triage_parser


def setup_issues_parser(parent: argparse.ArgumentParser) -> None:
    parent.set_defaults(func=lambda *args, **kwargs: parent.print_help())
    sp = parent.add_subparsers()

    setup_triage_parser(sp.add_parser("triage"))
