import argparse
from typing import Any, Optional, Callable
import logging

def parser_root(parent: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog=parent.prog, parents=[parent])

def instantiate(
    parent: argparse.ArgumentParser, 
    func: Optional[Callable[[Any], Any]] = None, 
    **kwargs: Callable[[argparse.ArgumentParser], argparse.ArgumentParser]
) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=parent.prog, parents=[parent], add_help=False)
    if func is None:
        func = lambda args: p.print_help()
    p.set_defaults(func=func)
    if len(kwargs) > 0:
        sp = p.add_subparsers()
        for k, f in kwargs.items():
            child = argparse.ArgumentParser(prog=' '.join([*parent.prog.split(), k]), parents=[parent], add_help=False)
            sp.add_parser(k, prog=' '.join([*parent.prog.split(), k]), parents=[f(child)])
    return p

def setup_logging(args: Any, name: str) -> None:
    logging.basicConfig()
    _l = logging.getLogger(name[:name.rfind('.')])
    _l.setLevel(args.log_level)
