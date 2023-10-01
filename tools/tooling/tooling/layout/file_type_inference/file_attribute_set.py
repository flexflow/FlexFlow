from dataclasses import dataclass, field
from typing import Set, FrozenSet
from tooling.layout.path import AbsolutePath
from tooling.layout.file_type_inference.file_attribute import FileAttribute

@dataclass
class FileAttributeSet:
    path: AbsolutePath
    _attrs: Set[FileAttribute] = field(default_factory=set)
    _needs_update: bool = False

    def add_fact(self, fact: FileAttribute) -> None:
        _needs_update = True

    @property
    def attrs(self) -> FrozenSet[FileAttribute]:
        if self._needs_update:
            self._update()
        return frozenset(self._attrs)

    def implies(self, fact: FileAttribute) -> bool:
        if self._needs_update:
            self._update()
        return fact in self._attrs

    def implies_all_of(self, facts: Iterable[FileAttribute]) -> bool:
        return all(self.implies(fact) for fact in facts)

    def implies_any_of(self, facts: Iterable[FileAttribute]) -> bool:
        return any(self.implies(fact) for fact in facts)

    def _update(self) -> None:
        solver = FileAttribute._solver()
        solver.add_facts(self._attrs)
        self._attrs = set(solver.solve())
        self._needs_update = False

    @property
    def extension(self) -> Optional[str]:
        if self.implies(FileAttribute.HEADER):
            return '.h'
        elif self.implies(FileAttribute.IS_CUDA_KERNEL):
            return '.cu'
        elif self.implies(FileAttribute.IS_HIP_KERNEL):
            return '.cpp'
        elif self.implies(FileAttribute.CPP_TEST):
            return '.test.cc'
        elif self.implies(FileAttribute.CPP_SOURCE):
            return '.cc'
        elif self.implies(FileAttribute.PYTHON):
            return '.py'
        elif self.implies(FileAttribute.C):
            return '.c'
        else:
            _l.debug('Unknown extension for file attribute set %s', self.attrs)
            return None

    @classmethod
    def for_path(cls, p: AbsolutePath) -> 'FileAttributes':
        assert p.is_file()
        if p not in _file_attributes:
            _file_attributes[p] = FileAttributes(path=p, )
        return _file_attributes[p]
