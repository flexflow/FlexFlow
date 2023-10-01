from tooling.layout.file_type_inference.solver import Solver
from enum import Enum, auto

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
    CPP_LIBRARY = auto()
    CMAKELISTS = auto() 
    CMAKE = auto()
    CPP_LIBRARY_CMAKELIST = auto()
    CPP_LIBRARY_SRC_DIR = auto()
    CPP_LIBRARY_INCLUDE_DIR = auto()
    CPP_LIBRARY_IN_SRC = auto()
    CPP_LIBRARY_IN_INCLUDE = auto()

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

