set(project_target kernels)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message("rocm path: $ENV{ROCM_PATH}")

project(${project_target} 
  LANGUAGES CXX HIP)

message("rocm path after: $ENV{ROCM_PATH}")


# if (DEFINED ENV{ROCM_PATH})
#   set(ROCM_PATH $ENV{ROCM_PATH})
# else()
  # message(FATAL_ERROR "ROCM_PATH is not set")
# endif()
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
if(CXX_IS_HIPCC)
  if(LINUX)
      if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
          message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
      endif()

      message(WARNING "Setting hipcc as the C++ compiler is legacy behavior."
              " Prefer setting the HIP compiler directly. See README for details.")
  endif()
else()
    # Forward AMDGPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
    if(AMDGPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES ${AMDGPU_ARGETS})
    endif()
    cmake_minimum_required(VERSION 3.21)
    enable_language(HIP)
endif()


find_package(hip REQUIRED)
find_package(miopen REQUIRED)
find_package(rccl REQUIRED)

file(GLOB_RECURSE SRC
     CONFIGURE_DEPENDS
     LIST_DIRECTORIES False
    #  src/*.cc
     src/hip/concat_kernels.cpp
     )

add_library(
  ${project_target}
  SHARED
  ${SRC}
)
target_include_directories(
  ${project_target}
  PRIVATE
    src/
  PUBLIC
    include/
)
target_link_libraries(
  ${project_target}
  op-attrs
  MIOpen
  hip::host
  rccl
)

define_ff_vars(${project_target})

set_target_properties(
  ${project_target} 
  PROPERTIES 
  HIP_STANDARD 17
)