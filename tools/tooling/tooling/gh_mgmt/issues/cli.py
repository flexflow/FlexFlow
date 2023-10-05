import argparse
from tooling.gh_mgmt.issues.triage.cli import get_triage_parser
from tooling.cli.parsing import instantiate 

def get_issues_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return instantiate(
        p,
        triage=get_triage_parser, 
    )
