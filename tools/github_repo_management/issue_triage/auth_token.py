import subprocess
import os
from typing import Optional

def get_auth_token() -> str:
    for method in [
        try_get_auth_token_from_env,
        try_get_auth_token_from_gh_cli
    ]:
        result = method()
        if result is not None:
            return result

    raise RuntimeError('Could not find auth token')

def try_get_auth_token_from_env() -> Optional[str]:
    return os.environ.get('TOKEN')

def try_get_auth_token_from_gh_cli() -> Optional[str]:
    try:
        return subprocess.check_output(['gh', 'auth', 'token']).decode().strip()
    except subprocess.CalledProcessError:
        return None
