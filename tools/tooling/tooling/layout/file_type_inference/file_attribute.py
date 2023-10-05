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
    BUILD_DIRECTORY = auto()
    COMPILE_COMMANDS_JSON = auto()
    CPP_FILE_GROUP_MEMBER = auto()
    CPP_LIBRARY_IS_VALID_FILE = auto()
    IN_CPP_LIBRARY = auto()
    CPP_FILE_GROUP_BASE = auto()

    # clang_format linter
    SUPPORTED_BY_CLANG_FORMAT = auto()
    WAS_PROPERLY_CLANG_FORMATTED = auto()
    WAS_IMPROPERLY_CLANG_FORMATTED = auto()
    IS_NOW_PROPERLY_CLANG_FORMATTED = auto()
    IS_NOW_IMPROPERLY_CLANG_FORMATTED = auto()
    DID_FIX_CLANG_FORMATTING = auto()

    HAS_UNCONVENTIONAL_INCLUDE_GUARDS = auto()
    SUPPORTED_BY_FIX_INCLUDE_GUARDS = auto()
    ORIGINALLY_HAD_CORRECT_INCLUDE_GUARD = auto()
    ORIGINALLY_HAD_INCORRECT_INCLUDE_GUARD = auto()
    NOW_HAS_CORRECT_INCLUDE_GUARD = auto()
    NOW_HAS_INCORRECT_INCLUDE_GUARD = auto()
    DID_FIX_INCLUDE_GUARD = auto()

    IS_SUPPORTED_BY_FIND_MISSING_FILES_LINTER = auto()
    ORIGINALLY_WAS_MISSING_HEADER_FILE = auto()
    ORIGINALLY_HAD_HEADER_FILE = auto()
    ORIGINALLY_WAS_MISSING_SOURCE_FILE = auto()
    ORIGINALLY_HAD_SOURCE_FILE = auto()
    ORIGINALLY_WAS_MISSING_TEST_FILE = auto()
    ORIGINALLY_HAD_TEST_FILE = auto()
    DID_FIX_MISSING_SOURCE_FILE = auto()
    NOW_IS_MISSING_HEADER_FILE = auto()
    NOW_HAS_HEADER_FILE = auto()
    NOW_IS_MISSING_SOURCE_FILE = auto()
    NOW_HAS_SOURCE_FILE = auto()
    NOW_IS_MISSING_TEST_FILE = auto()
    NOW_HAS_TEST_FILE = auto()

    HEADER = auto()
    IMPL = auto()

    PYTHON = auto()
    PYTHON_FF_TOOLS = auto()
    BASH_SCRIPT = auto()
    C = auto()

    IS_FFI_CODE = auto()
    EXTERNAL_FFI_HEADER = auto()
    INTERNAL_FFI_HEADER = auto()

    IS_INVALID_FILE = auto()
    IS_VALID_FILE = auto()

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return str(self.name)
