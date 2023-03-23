set(project_target kernels)

project(${project_target} 
  LANGUAGES CUDA)

file(GLOB_RECURSE SRC
     CONFIGURE_DEPENDS
     LIST_DIRECTORIES False
     src/cuda/*.cu src/*.cc)

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
  op-attrs
)

if (FF_GPU_BACKEND STREQUAL "cuda")
  target_compile_definitions(${project_target} PUBLIC FF_USE_CUDA)
elseif (FF_GPU_BACKEND STREQUAL "hip_cuda")
  target_compile_definitions(${project_target} PUBLIC FF_USE_HIP_CUDA)
elseif (FF_GPU_BACKEND STREQUAL "hip_rocm")
  target_compile_definitions(${project_target} PUBLIC FF_USE_HIP_ROCM)
endif()

target_compile_definitions(${project_target} PUBLIC MAX_TENSOR_DIM)

set_target_properties(
  ${project_target} 
  PROPERTIES 
  CUDA_STANDARD 11
)
