from dataclasses import dataclass
from tooling.gh_mgmt.cache.core.cache_entry import CacheEntry
from typing import Dict, Optional
from tooling.gh_mgmt.issues.triage.data_model.modification import Modification
from tooling.gh_mgmt.cache.core.previous_value_tag import PreviousValueTag 
from github import Github
from tooling.gh_mgmt.issues.triage.config import get_beginning_of_time
from datetime import timedelta
from tooling.json import Json
from tooling.gh_mgmt.issues.triage.data_model.issue_info import IssueInfo


@dataclass(frozen=True)
class IssueLastModificationsEntry(CacheEntry[Dict[int, Modification]]):
    repo_name: str

    def _cache_entry_name(self) -> str:
        return f"issues_last_modifications-{self.repo_name}"

    def _update(self, 
                new: Dict[int, Modification], 
                prev: PreviousValueTag[Optional[Dict[int, Modification]]]
                ) -> Dict[int, Modification]:
        assert prev.last_value is not None

        result: Dict[int, Modification] = {}
        key: int
        for key in set([*prev.last_value.keys(), *new.keys()]):
            prev_mod = prev.last_value.get(key, new.get(key))
            new_mod = new.get(key, prev.last_value.get(key))
            assert prev_mod is not None
            assert new_mod is not None
            latest_mod = max(prev_mod, new_mod)
            result[key] = latest_mod
        return result

    def _fetch_new(
        self, g: Github, prev: PreviousValueTag[Optional[Dict[int, Modification]]]
    ) -> Dict[int, Modification]:
        result: Dict[int, Modification] = {}
        repo = g.get_repo(self.repo_name)
        if prev.last_beginning_of_time > get_beginning_of_time():
            since = get_beginning_of_time()
        else:
            since = prev.last_run
        for comment in repo.get_issues_comments(since=since):
            current = Modification(
                user=comment.user.login,
                updated_at=comment.updated_at,
            )
            issue_id = IssueInfo.id_from_url(comment.issue_url)

            latest_modification = max(current, result.get(issue_id, current))
            result[issue_id] = latest_modification
        return result

    def _post_hook(self, t: Dict[int, Modification]) -> Dict[int, Modification]:
        return {k: v for k, v in t.items() if v.updated_at >= get_beginning_of_time()}

    def _default_value(self) -> Dict[int, Modification]:
        return {}

    def _ttl(self) -> timedelta:
        return timedelta(hours=1)

    def _to_jsonable(self, value: Dict[int, Modification]) -> Json:
        return {str(k): v.as_jsonable() for k, v in value.items()}

    def _from_jsonable(self, jsonable: Json) -> Dict[int, Modification]:
        assert isinstance(jsonable, dict)
        return {int(k): Modification.from_jsonable(v) for k, v in jsonable.items()}

    def _history_dependent(self) -> bool:
        return True
