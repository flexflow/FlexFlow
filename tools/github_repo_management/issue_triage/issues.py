from github import Github, Auth
from github.Repository import Repository
from github.NamedUser import NamedUser
from github.Organization import Organization

from datetime import timedelta, datetime, date
from frozendict import frozendict
from typing import List, Optional, Dict, Tuple, Set, Generic, TypeVar, FrozenSet, Callable, DefaultDict, Any, Type
from hashlib import md5
from pathlib import Path
from sqlitedict import SqliteDict
import functools
from collections import defaultdict
import logging
import dataclasses
from dataclasses import dataclass
import json
from abc import ABC, abstractmethod
from enum import Enum
from secrets import TOKEN

T = TypeVar('T')
Lazy = Callable[[], T]

NOW = datetime.now()

def to_datetime(date):
    return datetime.combine(date, datetime.min.time())

def ago(timedelta):
    return NOW - timedelta

def get_cache() -> SqliteDict:
    return SqliteDict('cache.sqlite', encode=json.dumps, decode=json.loads)

MAXIMUM_LOOKBACK = timedelta(days=365)
BEGINNING_OF_TIME = NOW - MAXIMUM_LOOKBACK 

@dataclass(frozen=True)
class PreviousValueTag(Generic[T]):
    last_run: datetime
    last_value: T
    last_beginning_of_time: datetime = BEGINNING_OF_TIME

    def as_jsonable(self, value_as_jsonable: Callable[[T], Any]) -> Any:
        return {
            'last_run' : self.last_run.isoformat(),
            'last_value' : value_as_jsonable(self.last_value),
            'last_beginning_of_time': self.last_beginning_of_time.isoformat()
        }

    @classmethod
    def from_jsonable(cls, value_from_jsonable: Callable[[Any], T], jsonable: Any) -> 'PreviousValueTag[T]':
        assert isinstance(jsonable, dict), type(jsonable)
        return cls(
            last_run=datetime.fromisoformat(jsonable['last_run']), 
            last_value=value_from_jsonable(jsonable['last_value']),
            last_beginning_of_time=datetime.fromisoformat(jsonable['last_beginning_of_time'])
        )

@dataclass(frozen=True, order=True)
class Modification:
    updated_at: datetime
    user: str

    def as_jsonable(self) -> Any:
        return {
            'updated_at': self.updated_at.isoformat(),
            'user': self.user
        }

    @classmethod
    def from_jsonable(cls, jsonable: Any) -> 'Modification':
        return cls(
            updated_at=datetime.fromisoformat(jsonable['updated_at']),
            user=jsonable['user']
        )

def id_from_url(url: str) -> int:
    return int(url.split('/')[-1])

class IssueState(Enum):
    open = 'open'
    closed = 'closed'
    all = 'all'

@dataclass(frozen=True)
class IssueInfo:
    assignees: FrozenSet[str]
    state: IssueState
    pull_request: Optional[int]
    closed_at: Optional[datetime]

    def as_jsonable(self) -> Any:
        return {
            'assignees': list(sorted(self.assignees)),
            'state': self.state.value,
            'pull_request': self.pull_request,
            'closed_at': self.closed_at.isoformat() if self.closed_at is not None else None
        }

    @classmethod
    def from_jsonable(cls, jsonable: Any) -> 'IssueInfo':
        return cls(
            assignees=frozenset(jsonable['assignees']),
            state=IssueState(jsonable['state']),
            pull_request=jsonable['pull_request'],
            closed_at=datetime.fromisoformat(jsonable['closed_at']) if jsonable['closed_at'] is not None else None
        )

    def closed_after(self, d: datetime) -> bool:
        if self.closed_at is None:
            return None
        else:
            assert self.state == IssueState.closed
            return self.closed_at >= d

    def closed_before(self, d: datetime) -> bool:
        if self.closed_at is None:
            return None
        else:
            assert self.state == IssueState.closed
            return self.closed_at <= d


class CacheEntry(ABC, Generic[T]):
    def _get_previous_value(self, cache=None) -> PreviousValueTag[Optional[T]]:
        if cache is None:
            cache = get_cache()
        cache_key = self._cache_entry_name()
        if cache_key in cache:
            return PreviousValueTag.from_jsonable(self._from_jsonable, cache[cache_key])
        else:
            return PreviousValueTag(last_run=ago(MAXIMUM_LOOKBACK), last_value=self._default_value())

    def _save_in_cache(self, value: Lazy[T], cache=None) -> T:
        if cache is None:
            cache = get_cache()
        prev = self._get_previous_value(cache)

        cache_expired = NOW - prev.last_run > self._ttl()
        lookback_changed = self._history_dependent() and prev.last_beginning_of_time > BEGINNING_OF_TIME
        if cache_expired or lookback_changed:
            cache_key = self._cache_entry_name()
            _l.info('Encountered cache miss on key: %s', cache_key)
            new_value = value()
            _l.debug('New value: %s', new_value)
            cache[cache_key] = PreviousValueTag(last_run=NOW, last_value=new_value).as_jsonable(lambda t: self._to_jsonable(t))
            cache.commit()
            return new_value
        else:
            assert prev.last_value is not None
            return prev.last_value

    def evaluate(self, g: Github) -> T:
        prev = self._get_previous_value()
        return self._post_hook(self._save_in_cache(
            lambda: self._update(self._fetch_new(g, prev), prev)
        ))

    def _post_hook(self, t: T) -> T:
        return t

    @abstractmethod
    def _cache_entry_name(self) -> str:
        pass

    @abstractmethod
    def _update(self, new: T, prev: PreviousValueTag[Optional[T]]) -> T:
        pass

    @abstractmethod
    def _default_value(self) -> Optional[T]:
        pass

    @abstractmethod
    def _fetch_new(self, g: Github, prev: PreviousValueTag[Optional[T]]) -> T:
        pass

    @abstractmethod
    def _ttl(self) -> timedelta:
        pass

    @abstractmethod
    def _to_jsonable(self, value: T) -> Any:
        pass

    @abstractmethod
    def _from_jsonable(self, jsonable: Any) -> T:
        pass

    @abstractmethod
    def _history_dependent(self) -> bool:
        pass

class StatelessCacheEntry(CacheEntry[T]):
    def _update(self, new: T, prev: PreviousValueTag[Optional[T]]) -> T:
        return new

    def _default_value(self) -> Optional[T]:
        return None

    def _fetch_new(self, g: Github, prev: PreviousValueTag[Optional[T]]) -> T:
        return self._fetch_stateless(g)

    @abstractmethod
    def _fetch_stateless(self, g: Github) -> T:
        pass

    def _history_dependent(self) -> bool:
        return False

@dataclass(frozen=True)
class OrgMembersEntry(StatelessCacheEntry[Set[str]]):
    org_name: str

    def _cache_entry_name(self) -> str:
        return f'org_members-{self.org_name}'

    def _fetch_stateless(self, g: Github) -> Set[str]:
        org = g.get_organization(self.org_name)
        return {m.login for m in org.get_members()}

    def _ttl(self) -> timedelta:
        return timedelta(hours=1)

    def _to_jsonable(self, value: Set[str]) -> Any:
        return list(sorted(value))

    def _from_jsonable(self, jsonable: Any) -> Set[str]:
        return set(jsonable)

@dataclass(frozen=True)
class RepoMembersEntry(StatelessCacheEntry[Set[str]]):
    repo_name: str
    
    def _cache_entry_name(self) -> str:
        return f'repo_members-{self.repo_name}'

    def _fetch_stateless(self, g: Github) -> Set[str]:
        repo = g.get_repo(self.repo_name)
        return {m.login for m in repo.get_collaborators()}

    def _ttl(self) -> timedelta:
        return timedelta(hours=1)

    def _to_jsonable(self, value: Set[str]) -> Any:
        return list(sorted(value))

    def _from_jsonable(self, jsonable: Any) -> Set[str]:
        return set(jsonable)

@dataclass(frozen=True)
class IssueInfoEntry(StatelessCacheEntry[IssueInfo]):
    repo_name: str
    issue_num: int

    def _cache_entry_name(self) -> str:
        return f'issue_info-{self.repo_name}_{self.issue_num}'

    def _fetch_stateless(self, g: Github) -> IssueInfo:
        repo = g.get_repo(self.repo_name)
        issue = repo.get_issue(self.issue_num)
        state = IssueState(issue.state)
        return IssueInfo(
            assignees=frozenset(a.login for a in issue.assignees),
            state=state,
            pull_request=id_from_url(issue.pull_request.html_url) if issue.pull_request is not None else None,
            closed_at=issue.closed_at if state == IssueState.closed else None
        )

    def _post_hook(self, t: IssueInfo) -> IssueInfo:
        if t.closed_before(BEGINNING_OF_TIME):
            return t.replace(closed_at=None)
        else:
            return t

    def _ttl(self) -> timedelta:
        return timedelta(hours=1)
    
    def _to_jsonable(self, value: IssueInfo) -> Any:
        return value.as_jsonable()

    def _from_jsonable(self, jsonable: Any) -> IssueInfo:
        return IssueInfo.from_jsonable(jsonable)

@dataclass(frozen=True)
class IssueLastModificationsEntry(CacheEntry[Dict[int, Modification]]):
    repo_name: str

    def _cache_entry_name(self) -> str:
        return f'issues_last_modifications-{self.repo_name}'

    def _update(self, new: Dict[int, Modification], prev: PreviousValueTag[Optional[Dict[int, Modification]]]):
        assert prev.last_value is not None

        result: Dict[int, Modification] = {}
        key: int
        for key in set([*prev.last_value.keys(), *new.keys()]):
            prev_mod = prev.last_value.get(key, new.get(key))
            new_mod = new.get(key, prev.last_value.get(key))
            latest_mod = max(prev_mod, new_mod)
            result[key] = latest_mod
        return result

    def _fetch_new(self, g: Github, prev: PreviousValueTag[Optional[Dict[int, Modification]]]) -> Dict[int, Modification]:
        result: Dict[int, Modification] = {}
        repo = g.get_repo(self.repo_name)
        if prev.last_beginning_of_time > BEGINNING_OF_TIME:
            since = BEGINNING_OF_TIME
        else:
            since = prev.last_run
        for comment in repo.get_issues_comments(since=since):
            current = Modification(
                user=comment.user.login, 
                updated_at=comment.updated_at,
            )
            issue_id = id_from_url(comment.issue_url)

            latest_modification = max(current, result.get(issue_id, current))
            result[issue_id] = latest_modification
        return result

    def _post_hook(self, t: Dict[int, Modification]) -> Dict[int, Modification]:
        return { k: v for k, v in t.items() if v.updated_at >= BEGINNING_OF_TIME }

    def _default_value(self):
        return {}

    def _ttl(self):
        return timedelta(hours=1)
    
    def _to_jsonable(self, value: Dict[int, Modification]) -> Any:
        return { str(k) : v.as_jsonable() for k, v in value.items() }

    def _from_jsonable(self, jsonable: Any) -> Dict[int, Modification]:
        return { int(k) : Modification.from_jsonable(v) for k, v in jsonable.items() }

    def _history_dependent(self) -> bool:
        return True

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

def get_unanswered_issues():
    auth = Auth.Token(TOKEN)
    c = Cache(github=Github(auth=auth))

    ff_org = 'flexflow'
    ff_repo = 'flexflow/FlexFlow'

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

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--json', action='store_true')
    p.add_argument('--level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='WARNING')
    args = p.parse_args()

    _l = logging.getLogger(__file__)
    _l.setLevel(getattr(logging, args.level))


    if args.json:
        print(json.dumps({
            str(k) : v.as_jsonable() 
            for k, v in get_unanswered_issues()
        }, indent=2))
    else:
        for issue_num, issue in get_unanswered_issues():
            assignees = ' '.join(list(sorted(issue.assignees)))
            print(f'{issue_num} ( {assignees} )')

