from tooling.layout.project import Project
from tooling.linting.framework.response import FixResponse
from typing import Optional

def check_unstaged_changes(project: Project, fix: bool, force: bool) -> Optional[FixResponse]:
    if fix:
        if len(project.get_unstaged_changes()) > 0 and not force:
            return FixResponse(did_succeed=False,
                message=(
                    'Refusing to modify files because there are unstaged changes in git.\n'
                    'If you\'re really sure you trust this tool to not break your changes, '
                    'you can override this message with --force.'
                )
            )
    return None
