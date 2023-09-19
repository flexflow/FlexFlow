import argparse
from .get_unanswered_issues import get_unanswered_issues

# @dataclass(frozen=True)
# def Config:
#     pass
# if args.json: print(json.dumps({
#         str(k) : v.as_jsonable()
#         for k, v in get_unanswered_issues()
#     }, indent=2))
# else:
#     for issue_num, issue in get_unanswered_issues():
#         assignees = ' '.join(list(sorted(issue.assignees)))
#         print(f'{issue_num} ( {assignees} )')


def main():
    print("triage!")
    get_unanswered_issues()


def setup_triage_parser(parent: argparse.ArgumentParser) -> None:
    p = parent
    p.add_argument("--time-window")
    p.set_defaults(func=main)
