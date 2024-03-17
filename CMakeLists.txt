cmake_minimum_required(VERSION 3.10)
project(FlexFlow)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

# set_property(CACHE FF_GPU_BACKEND PROPERTY STRINGS ${FF_GPU_BACKENDS})

include(CTest)
include(utils)
include(deps)

# TODO @lockshaw remove me
# https://discourse.nixos.org/t/get-clangd-to-find-standard-headers-in-nix-shell/11268/6
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

add_subdirectory(lib)
add_subdirectory(bin)
