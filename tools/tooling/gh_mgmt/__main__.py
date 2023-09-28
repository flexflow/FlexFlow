from .issues.cli import get_issues_parser
import argparse
import sys
import json
from .verbosity import add_verbosity_args, calculate_log_level
from .parsing import instantiate, parser_root

def main() -> int:
    PROGRAM_NAME = "repo"

    _p = argparse.ArgumentParser(prog=PROGRAM_NAME, add_help=False)
    add_verbosity_args(_p)

    p = parser_root(instantiate(
        _p,
        issues=get_issues_parser
    ))

    args = p.parse_args()
    args.log_level = calculate_log_level(args)

    result = args.func(args)
    if not args.silent and result is not None:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
