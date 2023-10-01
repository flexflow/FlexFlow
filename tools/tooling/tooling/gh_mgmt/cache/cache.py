from dataclasses import dataclass
from github import Github
from tooling.gh_mgmt.cache.entries.issue_info import IssueInfoEntry 
from tooling.gh_mgmt.cache.entries.repo_members import RepoMembersEntry 
from tooling.gh_mgmt.cache.entries.org_members import OrgMembersEntry
from tooling.gh_mgmt.cache.entries.issue_last_modifications import IssueLastModificationsEntry
from tooling.gh_mgmt.issues.triage.data_model.issue_info import IssueInfo 
from tooling.gh_mgmt.issues.triage.data_model.modification import Modification
from typing import Dict, Set


@dataclass
class Cache:
    github: Github

    def get_issue(self, repo: str, num: int) -> IssueInfo:
        entry = IssueInfoEntry(repo, num)
        return entry.evaluate(self.github)

    def get_repo_members(self, repo: str) -> Set[str]:
        entry = RepoMembersEntry(repo)
        return entry.evaluate(self.github)

    def get_org_members(self, org: str) -> Set[str]:
        entry = OrgMembersEntry(org)
        return entry.evaluate(self.github)

    def get_issue_last_modification_times(self, repo: str) -> Dict[int, Modification]:
        entry = IssueLastModificationsEntry(repo)
        return entry.evaluate(self.github)
