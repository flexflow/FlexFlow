cmake_minimum_required(VERSION 3.10)
project(FlexFlow)

set(
  CMAKE_MODULE_PATH 
  ${CMAKE_MODULE_PATH} 
  ${CMAKE_CURRENT_LIST_DIR}/cmake 
  ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules
)

# Detect OS type and Linux version (if it applies)
set(LINUX_VERSION "")
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  find_program(LSB_RELEASE_EXEC lsb_release)
  if(LSB_RELEASE_EXEC)
    execute_process(COMMAND ${LSB_RELEASE_EXEC} -r --short 
                    OUTPUT_VARIABLE LINUX_VERSION 
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Linux Version: ${LINUX_VERSION}")
  endif()
endif()

set(FF_MAX_DIM "5" CACHE STRING "Maximum tensor order")
set(FF_MAX_OPNAME "128" CACHE STRING "Maximum op name length")
set(FF_MAX_NUM_OUTPUTS "256" CACHE STRING "Maximum number of outputs (per operator)")
set(FF_MAX_NUM_INPUTS "256" CACHE STRING "Maximum number of inputs (per operator)")
set(FF_MAX_NUM_WEIGHTS "64" CACHE STRING "Maximum number of weights (per operator)")
set(FF_MAX_NUM_FUSED_OPERATORS "64" CACHE STRING "Maximum number of fused tensors")
set(FF_MAX_NUM_FUSED_TENSORS "64" CACHE STRING "Maximum number of input and output tensors per fused op")
set(FF_MAX_NUM_WORKERS "1024" CACHE STRING "Maximum number of GPUs")
set(FF_MAX_NUM_TASK_REGIONS "20" CACHE STRING 
  "Maximum number of regions that can be passed to a task through the TaskSpec interface")
set(FF_MAX_NUM_TASK_ARGUMENTS "5" CACHE STRING
  "Maximum number of arguments that can be declared in a TaskSignature")
option(FF_USE_NCCL "Run FlexFlow with NCCL" OFF)
option(FF_USE_PREBUILT_LEGION "Enable use of Legion pre-compiled library, if available" ON)
option(FF_USE_ALL_PREBUILT_LIBRARIES "Enable use of all pre-compiled libraries, if available" OFF)
option(FF_USE_PYTHON "Enable Python" ON)
option(FF_BUILD_FROM_PYPI "Build from pypi" OFF)
option(FF_USE_CODE_COVERAGE "Enable code coverage" OFF)

set(FF_GASNET_CONDUITS aries udp mpi ibv ucx)
set(FF_GASNET_CONDUIT "mpi" CACHE STRING "Select GASNet conduit ${FF_GASNET_CONDUITS}")
set_property(CACHE FF_GASNET_CONDUIT PROPERTY STRINGS ${FF_GASNET_CONDUITS})
set(FF_LEGION_NETWORKS "" CACHE STRING "Network backend(s) to use")

set(FF_GPU_BACKENDS cuda hip_cuda hip_rocm intel)
set(FF_GPU_BACKEND "cuda" CACHE STRING "Select GPU Backend ${FF_GPU_BACKENDS}")
set_property(CACHE FF_GPU_BACKEND PROPERTY STRINGS ${FF_GPU_BACKENDS})

option(FF_USE_EXTERNAL_LEGION "Use pre-installed Legion" OFF)
option(FF_USE_EXTERNAL_NCCL "Use pre-installed NCCL" OFF)
option(FF_USE_EXTERNAL_JSON "Use pre-installed nlohmann::json" OFF)
option(FF_USE_EXTERNAL_FMT "Use pre-installed fmt" OFF)
option(FF_USE_EXTERNAL_SPDLOG "Use pre-installed spdlog" OFF)
option(FF_USE_EXTERNAL_DOCTEST "Use pre-installed doctest" OFF)
option(FF_USE_EXTERNAL_RAPIDCHECK "Use pre-installed rapidcheck" OFF)
option(FF_USE_EXTERNAL_EXPECTED "Use pre-installed tl::expected" OFF)

option(FF_BUILD_RESNET "build resnet example" OFF)
option(FF_BUILD_RESNEXT "build resnext example" OFF)
option(FF_BUILD_ALEXNET "build alexnet example" OFF)
option(FF_BUILD_DLRM "build DLRM example" OFF)
option(FF_BUILD_XDL "build XDL example" OFF)
option(FF_BUILD_INCEPTION "build inception example" OFF)
option(FF_BUILD_CANDLE_UNO "build candle uno example" OFF)
option(FF_BUILD_TRANSFORMER "build transformer example" OFF)
option(FF_BUILD_MOE "build mixture of experts example" OFF)
option(FF_BUILD_MLP_UNIFY "build mlp unify example" OFF)
option(FF_BUILD_SPLIT_TEST "build split test example" OFF)
option(FF_BUILD_SPLIT_TEST_2 "build split test 2 example" OFF)
option(FF_BUILD_ALL_EXAMPLES "build all examples. Overrides others" OFF)
option(FF_BUILD_UNIT_TESTS "build non-operator unit tests" OFF)
option(FF_BUILD_SUBSTITUTION_TOOL "build substitution conversion tool" OFF)
option(FF_BUILD_VISUALIZATION_TOOL "build substitution visualization tool" ON)
option(FF_BUILD_ARG_PARSER "build command line argument parser" OFF)
option(FF_BUILD_BIN_EXPORT_MODEL_ARCH "build export-model-arch utility" ON)

set(FF_CUDA_ARCH "autodetect" CACHE STRING "Target CUDA Arch")
if (FF_CUDA_ARCH STREQUAL "")
  message(FATAL_ERROR "FF_CUDA_ARCH cannot be an empty string. Set it to `autodetect`, `all`, or pass one or multiple valid CUDA archs.")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(LIBEXT ".so")
endif()

include(cuda)
include(cudnn)
include(nccl)
if (FF_USE_CODE_COVERAGE)
  include(CodeCoverage)
  append_coverage_compiler_flags()
endif()
# set_property(CACHE FF_GPU_BACKEND PROPERTY STRINGS ${FF_GPU_BACKENDS})

include(json)
include(expected)
include(spdlog)
include(doctestlib) # named doctestlib to avoid a name collision with doctest.cmake in rapidcheck
include(visit_struct)
include(CTest)
include(fmt)
include(legion)
include(rapidcheck)
#include(gtest)

include(flexflow-utils)

# TODO @lockshaw remove me
# https://discourse.nixos.org/t/get-clangd-to-find-standard-headers-in-nix-shell/11268/6
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

add_subdirectory(lib)
add_subdirectory(bin)
