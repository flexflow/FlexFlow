set(project_target kernels)

project(${project_target} 
  LANGUAGES CUDA)

file(GLOB_RECURSE SRC
     CONFIGURE_DEPENDS
     LIST_DIRECTORIES False
     src/*.cc)

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
  cuda
  cudnn
  nccl
)

define_ff_vars(kernels)

set_target_properties(
  ${project_target} 
  PROPERTIES 
  CUDA_STANDARD 11
)
