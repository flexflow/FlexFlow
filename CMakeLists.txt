cmake_minimum_required(VERSION 3.10)
project(FlexFlow)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

set(FF_MAX_DIM "5" CACHE STRING "Maximum tensor order")
option(FF_USE_NCCL "Run FlexFlow with NCCL" OFF)
option(FF_USE_EXTERNAL_LEGION "Use pre-installed Legion" OFF)
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
option(FF_BUILD_VISUALIZATION_TOOL "build substitution visualization tool" OFF)
option(FF_BUILD_ARG_PARSER "build command line argument parser" OFF)

set(FLEXFLOW_LIBS "")
set(FLEXFLOW_INCLUDE_DIRS "")

include(json)
include(optional)
include(spdlog)
include(variant)
include(doctest)
include(CTest)
include(fmt)
#include(gtest)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(LIBEXT ".so")
endif()

# TODO @lockshaw remove me
# https://discourse.nixos.org/t/get-clangd-to-find-standard-headers-in-nix-shell/11268/6
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

add_subdirectory(lib)

add_library(substitution_loader SHARED
  ${CMAKE_SOURCE_DIR}/lib/ffc/src/substitution_loader.cc)
target_include_directories(substitution_loader PRIVATE ${FLEXFLOW_INCLUDE_DIRS})
target_link_libraries(substitution_loader nlohmann_json::nlohmann_json)

if(FF_BUILD_MOE OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/mixture_of_experts)
endif()

if(FF_BUILD_SUBSTITUTION_TOOL)
  add_subdirectory(bin/protobuf_to_json)
endif()

if(FF_BUILD_VISUALIZATION_TOOL)
  add_subdirectory(bin/substitutions_to_dot)
endif()

if(FF_BUILD_ARG_PARSER)
  add_subdirectory(bin/arg_parser)
endif()