set(project_target kernels)

project(${project_target} CUDA)

file(GLOB_RECURSE SRC
     CONFIGURE_DEPENDS
     LIST_DIRECTORIES False
     src/cuda/*.cu)

add_library(
  ${project_target}
  SHARED
  ${SRC}
)
target_include_directories(
  ${project_target}
  PRIVATE
    src/cuda/
  PUBLIC
    include/
)
target_link_libraries(
  ${project_target}
  op-meta
  ${LEGION_LIBRARY}
)

if (FF_GPU_BACKEND STREQUAL "cuda")
  target_compile_definitions(${project_target} PUBLIC FF_USE_CUDA)
elseif (FF_GPU_BACKEND STREQUAL "hip_cuda")
  target_compile_definitions(${project_target} PUBLIC FF_USE_HIP_CUDA)
elseif (FF_GPU_BACKEND STREQUAL "hip_rocm")
  target_compile_definitions(${project_target} PUBLIC FF_USE_HIP_ROCM)
endif()
