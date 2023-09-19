from .config.auth_token import get_auth_token
from github import Github
from github import Auth
from .cache import Cache
from datetime import timedelta


def get_unanswered_issues():
    auth = Auth.Token(get_auth_token())
    c = Cache(github=Github(auth=auth))

    ff_org = "flexflow"
    ff_repo = "flexflow/FlexFlow"

    members = c.get_org_members(ff_org) | c.get_repo_members(ff_repo)

    for issue_num, modification in c.get_issue_last_modification_times(ff_repo).items():
        if modification.user not in members:
            issue = c.get_issue(ff_repo, issue_num)
            if issue.pull_request == issue_num:
                continue
            elif issue.closed_after(modification.updated_at - timedelta(minutes=1)):
                continue
            else:
                yield issue_num, issue
