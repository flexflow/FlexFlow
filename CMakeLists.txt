cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(FlexFlow)

list(
  APPEND
  CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/cmake
  ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules
)

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
option(FF_USE_PREALM "Build with PRealm profiling interface" ON)
option(FF_USE_PYTHON "Enable Python" ON)
option(FF_BUILD_FROM_PYPI "Build from pypi" OFF)
option(FF_USE_CODE_COVERAGE "Enable code coverage" OFF)

set(FF_GPU_BACKENDS cuda hip_cuda hip_rocm intel)
set(FF_GPU_BACKEND "cuda" CACHE STRING "Select GPU Backend ${FF_GPU_BACKENDS}")
set_property(CACHE FF_GPU_BACKEND PROPERTY STRINGS ${FF_GPU_BACKENDS})

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
option(FF_BUILD_SP_IZATION_BENCHMARKING "build sp-ization benchmarking" ON)
option(FF_BUILD_ARG_PARSER "build command line argument parser" OFF)
option(FF_BUILD_BIN_COMPILE_MODEL "build compile-model binary" ON)
option(FF_BUILD_BIN_EXPORT_MODEL_ARCH "build export-model-arch utility" ON)
option(FF_BUILD_BIN_RUN_MODEL "build run-model binary" ON)

if (FF_USE_CODE_COVERAGE)
  include(CodeCoverage)
  append_coverage_compiler_flags()
endif()
# set_property(CACHE FF_GPU_BACKEND PROPERTY STRINGS ${FF_GPU_BACKENDS})

# Load after CodeCoverage because CodeCoverage blows up on NVCC compiler
include(cuda)
include(cudnn)
include(nccl)

include(json)
include(expected)
include(spdlog)
include(doctestlib) # named doctestlib to avoid a name collision with doctest.cmake in rapidcheck
include(gbenchmark)
include(libassert)
include(CTest)
include(fmt)
include(realm)
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
