from dataclasses import dataclass
from typing import Set
from tooling.gh_mgmt.cache.core.stateless_cache_entry import StatelessCacheEntry
from datetime import timedelta
from tooling.json import Json
from github import Github


@dataclass(frozen=True)
class OrgMembersEntry(StatelessCacheEntry[Set[str]]):
    org_name: str

    def _cache_entry_name(self) -> str:
        return f"org_members-{self.org_name}"

    def _fetch_stateless(self, g: Github) -> Set[str]:
        org = g.get_organization(self.org_name)
        return {m.login for m in org.get_members()}

    def _ttl(self) -> timedelta:
        return timedelta(hours=1)

    def _to_jsonable(self, value: Set[str]) -> Json:
        return list(sorted(value))

    def _from_jsonable(self, jsonable: Json) -> Set[str]:
        assert isinstance(jsonable, list)
        return set(jsonable)
