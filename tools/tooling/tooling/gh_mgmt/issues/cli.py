import argparse
from .triage.cli import get_triage_parser
from typing import Sequence
from ..parsing import instantiate


def get_issues_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return instantiate(
        p,
        triage=get_triage_parser, 
    )
