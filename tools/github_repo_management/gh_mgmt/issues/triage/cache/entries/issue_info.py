from dataclasses import dataclass
from ..core.stateless_cache_entry import StatelessCacheEntry
from ...data_model.issue_info import IssueInfo
from github import Github
from ...data_model.json import Json
from datetime import timedelta
from ...config.time import get_beginning_of_time
from ...data_model.issue_state import IssueState
import dataclasses


@dataclass(frozen=True)
class IssueInfoEntry(StatelessCacheEntry[IssueInfo]):
    repo_name: str
    issue_num: int

    def _cache_entry_name(self) -> str:
        return f"issue_info-{self.repo_name}_{self.issue_num}"

    def _fetch_stateless(self, g: Github) -> IssueInfo:
        repo = g.get_repo(self.repo_name)
        issue = repo.get_issue(self.issue_num)
        state = IssueState(issue.state)
        return IssueInfo(
            assignees=frozenset(a.login for a in issue.assignees),
            state=state,
            pull_request=IssueInfo.id_from_url(issue.pull_request.html_url) if issue.pull_request is not None else None,
            closed_at=issue.closed_at if state == IssueState.closed else None,
        )

    def _post_hook(self, t: IssueInfo) -> IssueInfo:
        if t.closed_before(get_beginning_of_time()):
            return dataclasses.replace(t, closed_at=None)
        else:
            return t

    def _ttl(self) -> timedelta:
        return timedelta(hours=1)

    def _to_jsonable(self, value: IssueInfo) -> Json:
        return value.as_jsonable()

    def _from_jsonable(self, jsonable: Json) -> IssueInfo:
        return IssueInfo.from_jsonable(jsonable)
