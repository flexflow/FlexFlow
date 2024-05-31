set(project_target kernels)

project(${project_target} 
	LANGUAGES CXX CUDA)

file(GLOB_RECURSE SRC
     CONFIGURE_DEPENDS
     LIST_DIRECTORIES False
     src/*.cc
     src/cuda/cuda_helper.cu
     src/cuda/ops/cast_kernels.cu
     src/cuda/ops/attention_kernels.cu
     src/cuda/ops/combine_kernels.cu
     src/cuda/ops/concat_kernels.cu
     src/cuda/ops/reshape_kernels.cu
     src/cuda/ops/partition_kernels.cu
     src/cuda/ops/replicate_kernels.cu
     src/cuda/ops/softmax_kernels.cu 
     src/cuda/ops/flat_kernels.cu
     src/cuda/ops/transpose_kernels.cu
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
  cuda
  cudnn
  nccl
  utils
)

define_ff_vars(${project_target})

set_target_properties(
  ${project_target} 
  PROPERTIES 
  CUDA_STANDARD 17
)

add_subdirectory(test)