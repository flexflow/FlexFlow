cmake_minimum_required(VERSION 3.10)

project(FlexFlowExample_AlexNet)
set(project_target alexnet)

set(CPU_SOURCES
  alexnet.cc
  alexnet.cu
  alexnet.h)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets)

cuda_add_executable(${project_target} ${CPU_SOURCES})
target_include_directories(${project_target} PRIVATE ${FLOW_INCLUDE} ${CMAKE_INSTALL_INCLUDEDIR})
#target_link_libraries(${project_target} flexflow ${Protobuf_LIBRARIES} cudnn cublas curand  "-L/sw/cuda/10.1/lib64 -L/scratch/shared/wwu12/cudnn/lib64 -lz -lrt -ldl -L/sw/cuda/10.1/lib64/stubs -lcuda" ${INSTALL_DIR}/lib64/liblegion.a ${INSTALL_DIR}/lib64/librealm.a)
target_link_libraries(${project_target} flexflow legion realm ${Protobuf_LIBRARIES} ${FLEXFLOW_EXT_LIBRARIES}  "-lz -lrt -ldl")
