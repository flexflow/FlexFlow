#! /usr/bin/env python3

from pathlib import Path

from lib.library_layout import (
    get_expected_source_paths,
)
from lib.
from lib.includes import (
    has_include
)

DIR = Path(__file__).parent

if __name__ == '__main__':
    RETURN_CODE = 0

    library_root = get_library_root(DIR)
    assert isinstance(library_root, Path)

    for src, includes in get_expected_source_paths(library_root).items():
        if not src.exists():
            pass
        
        # elif:
