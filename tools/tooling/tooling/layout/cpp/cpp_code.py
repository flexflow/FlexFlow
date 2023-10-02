from dataclasses import dataclass
from tooling.layout.path import AbsolutePath
from tooling.layout.cpp.library import Library, is_library_root 
from tooling.layout.file_type_inference.file_attribute import FileAttribute
from typing import Optional, FrozenSet

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tooling.layout.project import Project

@dataclass(frozen=True)
class CppCode:
    root_path: AbsolutePath
    project: 'Project'

    @property
    def blacklisted(self) -> FrozenSet[AbsolutePath]:
        return frozenset({
            self.lib_dir / 'compiler',
            self.lib_dir / 'ffi',
            self.lib_dir / 'kernels',
            self.lib_dir / 'op-attrs',
            self.lib_dir / 'pcg',
            self.lib_dir / 'runtime',
            self.lib_dir / 'substitutions'
        })

    @property
    def lib_dir(self) -> AbsolutePath:
        return self.project.lib_dir

    @property
    def clang_format_config_path(self) -> AbsolutePath:
        return self.root_path / '.clang-format-for-format-sh'

    @property
    def libraries(self) -> FrozenSet[Library]:
        return frozenset({
            Library.create(p, self) for p in self.project.file_types.with_attr(FileAttribute.CPP_LIBRARY)
        })

    def get_containing_library(self, path: AbsolutePath) -> Optional[Library]:
        for library in self.libraries:
            if library.contains(path):
                return library
        return None
