include(aliasing)

enable_language(CUDA)
find_package(CUDAToolkit 11.7 REQUIRED)

alias_library(deps::cublas CUDA::cublas)
