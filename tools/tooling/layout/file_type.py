from enum import Enum, auto
from typing import FrozenSet, TypeVar, Set, Optional, Iterator
from dataclasses import dataclass

T = TypeVar('T')

def union(it: Iterator[FrozenSet[T]]) -> Set[T]:
    result: Set[T] = set()
    result.union(set(v) for v in it)
    return result

class FileType(Enum):
    CPP = auto()
    CPP_HEADER = auto()
    CPP_PUBLIC_HEADER = auto()
    CPP_PRIVATE_HEADER = auto()
    CPP_CC = auto()
    CPP_SOURCE = auto()
    CPP_TEST = auto()
    CPP_FWDING_HEADER = auto()

    CPU_CPP = auto()
    CPU_CPP_HEADER = auto()
    CPU_CPP_PUBLIC_HEADER = auto()
    CPU_CPP_PRIVATE_HEADER = auto()
    CPU_CPP_CC = auto()
    CPU_CPP_SOURCE = auto()
    CPU_CPP_TEST = auto()
    CPU_CPP_FWDING_HEADER = auto()

    CUDA_CPP = auto()
    CUDA_CPP_HEADER = auto()
    CUDA_CPP_PUBLIC_HEADER = auto()
    CUDA_CPP_PRIVATE_HEADER = auto()
    CUDA_CPP_CC = auto()
    CUDA_CPP_SOURCE = auto()
    CUDA_CPP_TEST = auto()
    CUDA_CPP_FWDING_HEADER = auto()

    HIP_CPP = auto()
    HIP_CPP_HEADER = auto()
    HIP_CPP_PUBLIC_HEADER = auto()
    HIP_CPP_PRIVATE_HEADER = auto()
    HIP_CPP_CC = auto()
    HIP_CPP_SOURCE = auto()
    HIP_CPP_TEST = auto()
    HIP_CPP_FWDING_HEADER = auto()

    PYTHON = auto()
    PYTHON_FF_TOOLS = auto()
    PYTHON_FF_BINDINGS = auto()
    BASH_SCRIPT = auto()

    C = auto()
    C_HEADER = auto()
    C_SOURCE = auto()

    FFI_HEADER = auto()

    @staticmethod
    def _to_cpp(lhs: 'FileType') -> 'FileType':
        result: Optional[FileType] = None

        if lhs in [FileType.CPU_CPP, FileType.CUDA_CPP, FileType.HIP_CPP]:
            result = FileType.CPP
        elif lhs in [FileType.CPU_CPP_HEADER, FileType.CUDA_CPP_HEADER, FileType.HIP_CPP_HEADER]:
            result = FileType.CPP_HEADER
        elif lhs in [FileType.CPU_CPP_PUBLIC_HEADER, FileType.CUDA_CPP_PUBLIC_HEADER, FileType.HIP_CPP_PUBLIC_HEADER]:
            result = FileType.CPP_PUBLIC_HEADER
        elif lhs in [FileType.CPU_CPP_PRIVATE_HEADER, FileType.CUDA_CPP_PRIVATE_HEADER, FileType.HIP_CPP_PRIVATE_HEADER]:
            result = FileType.CPP_PRIVATE_HEADER
        elif lhs in [FileType.CPU_CPP_CC, FileType.CUDA_CPP_CC, FileType.HIP_CPP_CC]:
            result = FileType.CPP_CC
        elif lhs in [FileType.CPU_CPP_SOURCE, FileType.CUDA_CPP_SOURCE, FileType.HIP_CPP_SOURCE]:
            result = FileType.CPP_SOURCE
        elif lhs in [FileType.CPU_CPP_TEST, FileType.CUDA_CPP_TEST, FileType.HIP_CPP_TEST]:
            result = FileType.CPP_TEST
        elif lhs in [FileType.CPU_CPP_FWDING_HEADER, FileType.CUDA_CPP_FWDING_HEADER, FileType.HIP_CPP_FWDING_HEADER]:
            result = FileType.CPP_FWDING_HEADER

        if result is None:
            raise ValueError(f'Unhandled file type {lhs}')
        else:
            return result

    @staticmethod
    def _to_cuda_cpp(lhs: 'FileType') -> 'FileType':
        for ft in FileType._cuda_cpp():
            if lhs == FileType._to_cpp(ft):
                return FileType._to_cpp(ft)
        raise ValueError(f'Unhandled file type {lhs}')

    @staticmethod
    def _to_hip_cpp(lhs: 'FileType') -> 'FileType':
        for ft in FileType._hip_cpp():
            if lhs == FileType._to_cpp(ft):
                return FileType._to_cpp(ft)
        raise ValueError(f'Unhandled file type {lhs}')

    @staticmethod
    def _to_cpu_cpp(lhs: 'FileType') -> 'FileType':
        for ft in FileType._cpu_cpp():
            if lhs == FileType._to_cpp(ft):
                return FileType._to_cpp(ft)
        raise ValueError(f'Unhandled file type {lhs}')

    @staticmethod
    def _to_cpp_type(exact_ty: 'FileType', cpp_type: 'FileType') -> 'FileType':
        if cpp_type == FileType.CUDA_CPP:
            return FileType._to_cuda_cpp(cpp_type)
        elif cpp_type == FileType.HIP_CPP:
            return FileType._to_hip_cpp(cpp_type)
        elif cpp_type == FileType.CPU_CPP:
            return FileType._to_cpu_cpp(cpp_type)
        else:
            raise ValueError(f'Invalid cpp type {cpp_type}')

    @staticmethod
    def _weaken_cpp_filetype(lhs: 'FileType') -> FrozenSet['FileType']:
        result: Optional[FileType] = None
        if lhs in [FileType.CPP_HEADER, FileType.CPP_CC]:
            result = FileType.CPP
        elif lhs in [FileType.CPP_PUBLIC_HEADER, FileType.CPP_PRIVATE_HEADER, FileType.CPP_FWDING_HEADER]:
            result = FileType.CPP_HEADER
        elif lhs in [FileType.CPP_SOURCE, FileType.CPP_TEST]:
            result = FileType.CPP_CC
        
        if result is None:
            return frozenset()
        else:
            return frozenset({result})

    @staticmethod
    def _cpp() -> FrozenSet['FileType']:
        return frozenset({
            FileType.CPP,
            FileType.CPP_HEADER,
            FileType.CPP_PUBLIC_HEADER,
            FileType.CPP_PRIVATE_HEADER,
            FileType.CPP_CC,
            FileType.CPP_SOURCE,
            FileType.CPP_TEST,
            FileType.CPP_FWDING_HEADER,
        })

    @staticmethod
    def _cpu_cpp() -> FrozenSet['FileType']:
        return frozenset(union(map(FileType._to_cpu_cpp, FileType._cpp())))

    @staticmethod
    def _cuda_cpp() -> FrozenSet['FileType']:
        return frozenset(union(map(FileType._to_cuda_cpp, FileType._cpp())))

    @staticmethod
    def _hip_cpp() -> FrozenSet['FileType']:
        return frozenset(union(map(FileType._to_hip_cpp, FileType._cpp())))

    @staticmethod
    def _python() -> FrozenSet['FileType']:
        return frozenset({
            FileType.PYTHON,
            FileType.PYTHON_FF_BINDINGS,
            FileType.PYTHON_FF_TOOLS
        })

    @staticmethod
    def _c() -> FrozenSet['FileType']:
        return frozenset({
            FileType.C,
            FileType.C_HEADER,
            FileType.C_SOURCE
        })

    def implies(self, other: 'FileType') -> bool:
        return other in FileType.implications(self)

    @staticmethod
    def _implies(lhs: 'FileType') -> FrozenSet['FileType']:
        if lhs in FileType._cpp():
            return FileType._weaken_cpp_filetype(lhs)
        elif lhs in (FileType._cpu_cpp() | FileType._hip_cpp() | FileType._cuda_cpp()):
            return FileType._weaken_cpp_filetype(lhs) | FileType._to_cpp(lhs)
        elif lhs in FileType._python():
            return frozenset({FileType.PYTHON})
        elif lhs in FileType._c():
            return frozenset({FileType.C})
        elif lhs == FileType.FFI_HEADER:
            return frozenset({FileType.C_HEADER, FileType.CPU_CPP_PUBLIC_HEADER})
        elif lhs == FileType.BASH_SCRIPT:
            return frozenset()
        else:
            raise ValueError(f'Unhandled file type {lhs}')

    @staticmethod
    def implications(lhs: 'FileType') -> FrozenSet['FileType']:
        rhs: FrozenSet[FileType] = frozenset({lhs})
        while True:
            new_rhs = frozenset(union(FileType._implies(ft) for ft in rhs)) | rhs
            if new_rhs == rhs:
                return rhs
            else:
                rhs = new_rhs


@dataclass(frozen=True)
class FileTypes:
    types: FrozenSet[FileType]
