from tooling.gh_mgmt.issues.triage.config import get_auth_token
from github import Github
from github import Auth
from tooling.gh_mgmt.cache.cache import Cache
from datetime import timedelta
from tooling.gh_mgmt.issues.triage.data_model.issue_info import IssueInfo
from typing import Mapping


def get_unanswered_issues() -> Mapping[int, IssueInfo]:
    auth = Auth.Token(get_auth_token())
    c = Cache(github=Github(auth=auth))

    ff_org = "flexflow"
    ff_repo = "flexflow/FlexFlow"

    members = c.get_org_members(ff_org) | c.get_repo_members(ff_repo)

    result = {}
    for issue_num, modification in c.get_issue_last_modification_times(ff_repo).items():
        if modification.user not in members:
            issue = c.get_issue(ff_repo, issue_num)
            if issue.pull_request == issue_num:
                continue
            elif issue.closed_after(modification.updated_at - timedelta(minutes=1)):
                continue
            else:
                result[issue_num] = issue
    return result
