from dataclasses import dataclass
from ..core.stateless_cache_entry import StatelessCacheEntry
from typing import Set, Any
from github import Github
from datetime import timedelta


@dataclass(frozen=True)
class RepoMembersEntry(StatelessCacheEntry[Set[str]]):
    repo_name: str

    def _cache_entry_name(self) -> str:
        return f"repo_members-{self.repo_name}"

    def _fetch_stateless(self, g: Github) -> Set[str]:
        repo = g.get_repo(self.repo_name)
        return {m.login for m in repo.get_collaborators()}

    def _ttl(self) -> timedelta:
        return timedelta(hours=1)

    def _to_jsonable(self, value: Set[str]) -> Any:
        return list(sorted(value))

    def _from_jsonable(self, jsonable: Any) -> Set[str]:
        return set(jsonable)
