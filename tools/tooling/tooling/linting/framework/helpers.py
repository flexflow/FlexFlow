from tooling.layout.project import Project
from tooling.linting.framework.response import ErrorResponse
from tooling.layout.path import AbsolutePath
from typing import Optional, List, Iterable
from tooling.layout.file_type_inference.file_attribute import FileAttribute


def check_unstaged_changes(project: Project, fix: bool, force: bool) -> Optional[ErrorResponse]:
    if fix:
        if len(project.get_unstaged_changes()) > 0 and not force:
            return ErrorResponse(
                message=(
                    "Refusing to modify files because there are unstaged changes in git.\n"
                    "If you're really sure you trust this tool to not break your changes, "
                    "you can override this message with --force."
                )
            )
    return None


def jsonify_files(project: Project, files: Iterable[AbsolutePath]) -> List[str]:
    return list(sorted(str(p.relative_to(project.root_path)) for p in files))


def jsonify_files_with_attr(project: Project, attr: FileAttribute) -> List[str]:
    return jsonify_files(project, project.file_types.with_attr(attr))
