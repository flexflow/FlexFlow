cmake_minimum_required(VERSION 3.1)

project(FlexFlowExample_PCA)
set(project_target pca)


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets)

cuda_add_executable(${project_target} pca.cc)
target_include_directories(${project_target} PRIVATE ${FLOW_INCLUDE} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(${project_target} flexflow)
