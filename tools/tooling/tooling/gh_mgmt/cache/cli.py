from tooling.cli.parsing import instantiate, setup_logging
import argparse
from typing import Any
import logging


def main_clear_cache(args: Any) -> None:
    setup_logging(args, __name__)
    _l = logging.getLogger(__name__)


def get_clear_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return instantiate(p, func=main_clear_cache)


def get_cache_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return instantiate(p, clear=get_clear_parser)
