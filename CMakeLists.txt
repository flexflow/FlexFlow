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
option(FF_BUILD_MLP_UNIFY "build mlp unify example" OFF)
option(FF_BUILD_SPLIT_TEST "build split test example" OFF)
option(FF_BUILD_SPLIT_TEST_2 "build split test 2 example" OFF)
option(FF_BUILD_ALL_EXAMPLES "build all examples. Overrides others" OFF)
option(FF_BUILD_UNIT_TESTS "build non-operator unit tests" OFF)
option(FF_BUILD_SUBSTITUTION_TOOL "build substitution conversion tool" OFF)
option(FF_BUILD_VISUALIZATION_TOOL "build substitution visualization tool" OFF)

include(json)
include(optional)
include(spdlog)
include(variant)
include(fmt)

# TODO @lockshaw remove me
# https://discourse.nixos.org/t/get-clangd-to-find-standard-headers-in-nix-shell/11268/6
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

add_subdirectory(lib)
