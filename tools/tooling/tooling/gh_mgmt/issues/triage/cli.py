import argparse
from tooling.gh_mgmt.issues.triage.get_unanswered_issues import get_unanswered_issues 
from pathlib import Path
from datetime import timedelta
from typing import Any, Optional
from tooling.gh_mgmt.issues.triage.argparse_env import EnvStore
from tooling.gh_mgmt.issues.triage.config import get_default_cache_dir, get_auth_token_from_gh_cli, init_config, get_cache_path
import re
from tooling.json import Json
import logging
from tooling.cli.parsing import instantiate 

def main(raw_args: Any) -> Optional[Json]:
    logging.basicConfig()
    _l = logging.getLogger(__name__)
    _l.setLevel(raw_args.log_level)

    auth_token = raw_args.auth_token
    if auth_token is None:
        auth_token = get_auth_token_from_gh_cli()
    if auth_token is None:
        raise RuntimeError('Could not find github auth token')

    init_config(
        max_lookback=raw_args.max_lookback,
        cache_dir=raw_args.cache_dir,
        auth_token=auth_token
    )

    if False:
        cache_path = get_cache_path()
        _l.info(f'Removing cache at {cache_path}')
        cache_path.unlink(missing_ok=True)
    else:
        return {str(k) : v.as_jsonable() for k, v in get_unanswered_issues().items()}

def clear_cache(raw_args: Any) -> None:
    logging.basicConfig()
    _l = logging.getLogger(__name__)
    _l.setLevel(raw_args.log_level)
    init_config(cache_dir=raw_args.cache_dir)
    cache_path = get_cache_path()
    _l.info(f'Removing cache at {cache_path}')
    try:
        cache_path.unlink()
    except FileNotFoundError:
        _l.debug(f'Could not find cache file at {cache_path} to unlink')

def parse_timedelta(s: str) -> timedelta:
    m = re.match(r'(?P<number>\d+)(?P<period>[dmyDMY])', s)
    if m is None:
        raise argparse.ArgumentTypeError(f'Invalid timedelta {s}')
    
    number = int(m.group('number'))
    period = m.group('period').lower()
    if period == 'd':
        return timedelta(days=number)
    elif period == 'm':
        return timedelta(days=number * 31)
    else:
        assert period == 'y'
        return timedelta(days=number * 365)

def get_clear_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return instantiate(p, func=clear_cache)

def get_triage_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument('--max-lookback', type=parse_timedelta, action=EnvStore, env_sources=["FFGH_MAX_LOOKBACK"], default='1y')
    p.add_argument('--cache-dir', type=Path, action=EnvStore, env_sources=["FFGH_CACHE_DIR"], default=get_default_cache_dir())
    p.add_argument('--auth-token', type=str, action=EnvStore, env_sources=["FFGH_AUTH_TOKEN"], default=None)
    
    return instantiate(
        p, 
        func=main,
        clear=get_clear_parser,
    )
