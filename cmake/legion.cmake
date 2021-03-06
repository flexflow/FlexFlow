set(LEGION_ROOT ${CMAKE_SOURCE_DIR}/deps/legion)
if(FF_USE_PYTHON)
  set(Legion_USE_Python ON CACHE BOOL "enable Legion_USE_Python")
endif()
set(Legion_MAX_DIM ${MAX_DIM} CACHE STRING "Maximum number of dimensions")
set(Legion_USE_CUDA ON CACHE BOOL "enable Legion_USE_CUDA")
set(Legion_CUDA_ARCH ${FF_CUDA_ARCH} CACHE STRING "Legion CUDA ARCH")
add_subdirectory(deps/legion)
list(APPEND FLEXFLOW_INCLUDE_DIRS
  ${LEGION_ROOT}/runtime
  ${LEGION_ROOT}/runtime/mappers
  ${LEGION_ROOT}/runtime/realm/transfer
  ${CMAKE_CURRENT_BINARY_DIR}/deps/legion/runtime)
list(APPEND FLEXFLOW_EXT_LIBRARIES
  Legion)
