from enum import Enum, auto
from typing import FrozenSet, TypeVar, Set, Optional, Iterator, Generic, Iterable, Tuple, Dict
from dataclasses import dataclass, field
from tooling.layout.path import AbsolutePath
import logging

_l = logging.getLogger(__name__)

T = TypeVar('T')

def union(it: Iterator[FrozenSet[T]]) -> Set[T]:
    result: Set[T] = set()
    result.union(set(v) for v in it)
    return result

@dataclass(frozen=True)
class Rule(Generic[T]):
    pre_condition: FrozenSet[T]
    post_condition: T

@dataclass
class Solver(Generic[T]):
    _rules: Set[Rule[T]] = field(default_factory=set)

    def add_conjunction_rule(self, rhs: T, lhs: Iterable[T]) -> None:
        self._rules.add(Rule(pre_condition=frozenset(lhs), post_condition=rhs))

    def add_conjunction_rules(self, rules: Iterable[Tuple[T, Iterable[T]]]) -> None:
        for post, pre in rules:
            self.add_conjunction_rule(post, pre)

    def add_disjunction_rule(self, rhs: T, lhs: Iterable[T]) -> None:
        for ll in lhs:
            self.add_conjunction_rule(rhs, [ll])

    def add_disjunction_rules(self, rules: Iterable[Tuple[T, Iterable[T]]]) -> None:
        for post, pre in rules:
            self.add_disjunction_rule(post, pre)

    def add_alias(self, lhs: Iterable[T], rhs: Iterable[T]) -> None:
        lhs = set(lhs)
        rhs = set(rhs)
        for ll in lhs:
            self.add_conjunction_rule(ll, rhs)
        for rr in rhs:
            self.add_conjunction_rule(rr, lhs)

    def add_aliases(self, aliases: Iterable[Tuple[Iterable[T], Iterable[T]]]) -> None:
        for aa in aliases:
            self.add_alias(*aa)
        
    def add_fact(self, fact: T) -> None:
        self.add_conjunction_rule(fact, [])

    def add_facts(self, facts: Iterable[T]) -> None:
        for fact in facts:
            self.add_fact(fact)

    def solve(self) -> FrozenSet[T]:
        solution: Set[T] = set()
        old_solution: Optional[Set[T]] = None

        rules = set(self._rules)

        while solution != old_solution:
            old_solution = set(solution)
            for rule in list(rules):
                if rule.post_condition in old_solution:
                    rules.remove(rule)
                elif all(pre in old_solution for pre in rule.pre_condition):
                    solution.add(rule.post_condition)

        return frozenset(solution)

class FileAttribute(Enum):
    CPP = auto()
    CPP_PUBLIC_HEADER = auto()
    CPP_PRIVATE_HEADER = auto()
    CPP_SOURCE = auto()
    CPP_TEST = auto()
    CPP_FWDING_HEADER = auto()
    IS_NOT_KERNEL = auto()
    IS_CPU_KERNEL = auto()
    IS_CUDA_KERNEL = auto()
    IS_HIP_KERNEL = auto()

    HEADER = auto()
    IMPL = auto()

    PYTHON = auto()
    PYTHON_FF_TOOLS = auto()
    BASH_SCRIPT = auto()
    C = auto()

    IS_FFI_CODE = auto()
    EXTERNAL_FFI_HEADER = auto()
    INTERNAL_FFI_HEADER = auto()

    @staticmethod
    def _solver() -> Solver['FileAttribute']:
        solver: Solver['FileAttribute'] = Solver()

        solver.add_disjunction_rules([
            (
                FileAttribute.CPP, 
                (
                    FileAttribute.CPP_PUBLIC_HEADER, 
                    FileAttribute.CPP_PRIVATE_HEADER, 
                    FileAttribute.CPP_FWDING_HEADER, 
                    FileAttribute.CPP_TEST, 
                    FileAttribute.IS_CPU_KERNEL,
                    FileAttribute.IS_CUDA_KERNEL,
                    FileAttribute.IS_HIP_KERNEL,
                )
            ),
            (
                FileAttribute.CPP_PRIVATE_HEADER,
                (
                    FileAttribute.INTERNAL_FFI_HEADER,
                )
            ),
            (
                FileAttribute.C,
                (
                    FileAttribute.EXTERNAL_FFI_HEADER,
                )
            ),
            (
                FileAttribute.IMPL, 
                (
                    FileAttribute.CPP_SOURCE, 
                    FileAttribute.CPP_TEST,
                )
            ),
            (
                FileAttribute.PYTHON,
                (
                    FileAttribute.PYTHON_FF_TOOLS,
                )
            ),
        ])

        return solver

    # @staticmethod
    # def _to_cpp(lhs: 'FileAttribute') -> 'FileAttribute':
    #     result: Optional[FileAttribute] = None

    #     if lhs in [FileAttribute.CPU_CPP, FileAttribute.CUDA_CPP, FileAttribute.HIP_CPP]:
    #         result = FileAttribute.CPP
    #     elif lhs in [FileAttribute.CPU_CPP_HEADER, FileAttribute.CUDA_CPP_HEADER, FileAttribute.HIP_CPP_HEADER]:
    #         result = FileAttribute.CPP_HEADER
    #     elif lhs in [FileAttribute.CPU_CPP_PUBLIC_HEADER, FileAttribute.CUDA_CPP_PUBLIC_HEADER, FileAttribute.HIP_CPP_PUBLIC_HEADER]:
    #         result = FileAttribute.CPP_PUBLIC_HEADER
    #     elif lhs in [FileAttribute.CPU_CPP_PRIVATE_HEADER, FileAttribute.CUDA_CPP_PRIVATE_HEADER, FileAttribute.HIP_CPP_PRIVATE_HEADER]:
    #         result = FileAttribute.CPP_PRIVATE_HEADER
    #     elif lhs in [FileAttribute.CPU_CPP_CC, FileAttribute.CUDA_CPP_CC, FileAttribute.HIP_CPP_CC]:
    #         result = FileAttribute.CPP_CC
    #     elif lhs in [FileAttribute.CPU_CPP_SOURCE, FileAttribute.CUDA_CPP_SOURCE, FileAttribute.HIP_CPP_SOURCE]:
    #         result = FileAttribute.CPP_SOURCE
    #     elif lhs in [FileAttribute.CPU_CPP_TEST, FileAttribute.CUDA_CPP_TEST, FileAttribute.HIP_CPP_TEST]:
    #         result = FileAttribute.CPP_TEST
    #     elif lhs in [FileAttribute.CPU_CPP_FWDING_HEADER, FileAttribute.CUDA_CPP_FWDING_HEADER, FileAttribute.HIP_CPP_FWDING_HEADER]:
    #         result = FileAttribute.CPP_FWDING_HEADER

    #     if result is None:
    #         raise ValueError(f'Unhandled file type {lhs}')
    #     else:
    #         return result

    # @staticmethod
    # def _to_cuda_cpp(lhs: 'FileAttribute') -> 'FileAttribute':
    #     for ft in FileAttribute._cuda_cpp():
    #         if lhs == FileAttribute._to_cpp(ft):
    #             return FileAttribute._to_cpp(ft)
    #     raise ValueError(f'Unhandled file type {lhs}')

    # @staticmethod
    # def _to_hip_cpp(lhs: 'FileAttribute') -> 'FileAttribute':
    #     for ft in FileAttribute._hip_cpp():
    #         if lhs == FileAttribute._to_cpp(ft):
    #             return FileAttribute._to_cpp(ft)
    #     raise ValueError(f'Unhandled file type {lhs}')

    # @staticmethod
    # def _to_cpu_cpp(lhs: 'FileAttribute') -> 'FileAttribute':
    #     for ft in FileAttribute._cpu_cpp():
    #         if lhs == FileAttribute._to_cpp(ft):
    #             return FileAttribute._to_cpp(ft)
    #     raise ValueError(f'Unhandled file type {lhs}')

    # @staticmethod
    # def _to_cpp_type(exact_ty: 'FileAttribute', cpp_type: 'FileAttribute') -> 'FileAttribute':
    #     if cpp_type == FileAttribute.CUDA_CPP:
    #         return FileAttribute._to_cuda_cpp(cpp_type)
    #     elif cpp_type == FileAttribute.HIP_CPP:
    #         return FileAttribute._to_hip_cpp(cpp_type)
    #     elif cpp_type == FileAttribute.CPU_CPP:
    #         return FileAttribute._to_cpu_cpp(cpp_type)
    #     else:
    #         raise ValueError(f'Invalid cpp type {cpp_type}')

    # @staticmethod
    # def _weaken_cpp_filetype(lhs: 'FileAttribute') -> FrozenSet['FileAttribute']:
    #     result: Optional[FileAttribute] = None
    #     if lhs in [FileAttribute.CPP_HEADER, FileAttribute.CPP_CC]:
    #         result = FileAttribute.CPP
    #     elif lhs in [FileAttribute.CPP_PUBLIC_HEADER, FileAttribute.CPP_PRIVATE_HEADER, FileAttribute.CPP_FWDING_HEADER]:
    #         result = FileAttribute.CPP_HEADER
    #     elif lhs in [FileAttribute.CPP_SOURCE, FileAttribute.CPP_TEST]:
    #         result = FileAttribute.CPP_CC
        
    #     if result is None:
    #         return frozenset()
    #     else:
    #         return frozenset({result})

    # @staticmethod
    # def _cpp() -> FrozenSet['FileAttribute']:
    #     return frozenset({
    #         FileAttribute.CPP,
    #         FileAttribute.CPP_HEADER,
    #         FileAttribute.CPP_PUBLIC_HEADER,
    #         FileAttribute.CPP_PRIVATE_HEADER,
    #         FileAttribute.CPP_CC,
    #         FileAttribute.CPP_SOURCE,
    #         FileAttribute.CPP_TEST,
    #         FileAttribute.CPP_FWDING_HEADER,
    #     })

    # @staticmethod
    # def _cpu_cpp() -> FrozenSet['FileAttribute']:
    #     return frozenset(map(FileAttribute._to_cpu_cpp, FileAttribute._cpp()))

    # @staticmethod
    # def _cuda_cpp() -> FrozenSet['FileAttribute']:
    #     return frozenset(map(FileAttribute._to_cuda_cpp, FileAttribute._cpp()))

    # @staticmethod
    # def _hip_cpp() -> FrozenSet['FileAttribute']:
    #     return frozenset(map(FileAttribute._to_hip_cpp, FileAttribute._cpp()))

    # @staticmethod
    # def _python() -> FrozenSet['FileAttribute']:
    #     return frozenset({
    #         FileAttribute.PYTHON,
    #         FileAttribute.PYTHON_FF_BINDINGS,
    #         FileAttribute.PYTHON_FF_TOOLS
    #     })

    # @staticmethod
    # def _c() -> FrozenSet['FileAttribute']:
    #     return frozenset({
    #         FileAttribute.C,
    #         FileAttribute.C_HEADER,
    #         FileAttribute.C_SOURCE
    #     })

    # def implies(self, other: 'FileAttribute') -> bool:
    #     return other in FileAttribute.implications(self)

    # @staticmethod
    # def _implies(lhs: 'FileAttribute') -> FrozenSet['FileAttribute']:
    #     if lhs in FileAttribute._cpp():
    #         return FileAttribute._weaken_cpp_filetype(lhs)
    #     elif lhs in (FileAttribute._cpu_cpp() | FileAttribute._hip_cpp() | FileAttribute._cuda_cpp()):
    #         return FileAttribute._weaken_cpp_filetype(lhs) | {FileAttribute._to_cpp(lhs)}
    #     elif lhs in FileAttribute._python():
    #         return frozenset({FileAttribute.PYTHON})
    #     elif lhs in FileAttribute._c():
    #         return frozenset({FileAttribute.C})
    #     elif lhs == FileAttribute.FFI_HEADER:
    #         return frozenset({FileAttribute.C_HEADER, FileAttribute.CPU_CPP_PUBLIC_HEADER})
    #     elif lhs == FileAttribute.BASH_SCRIPT:
    #         return frozenset()
    #     else:
    #         raise ValueError(f'Unhandled file type {lhs}')

    # @staticmethod
    # def implications(lhs: 'FileAttribute') -> FrozenSet['FileAttribute']:
    #     rhs: FrozenSet[FileAttribute] = frozenset({lhs})
    #     while True:
    #         new_rhs = frozenset(union(FileAttribute._implies(ft) for ft in rhs)) | rhs
    #         if new_rhs == rhs:
    #             return rhs
    #         else:
    #             rhs = new_rhs

_SOLVER = FileAttribute._solver()

_file_attributes: Dict[AbsolutePath, 'FileAttributes'] = {}

@dataclass
class FileAttributes:
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
