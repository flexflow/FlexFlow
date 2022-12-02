cmake_minimum_required(VERSION 3.10)

project(FlexFlowExampleInference_MLPUnify)
set(project_target mlp_inference_unify)

set(CPU_SRC
  ${FLEXFLOW_CPP_DRV_SRC}
  mlp.cc
  mlp.h
  ../data_generator.h)

set(GPU_SRC alexnet.cu)

cuda_add_executable(${project_target} ${CPU_SRC} ${GPU_SRC})
target_include_directories(${project_target} PRIVATE ${CMAKE_SOURCE_DIR}/examples/cpp/inference)
target_include_directories(${project_target} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(${project_target} -Wl,--whole-archive flexflow -Wl,--no-whole-archive ${FLEXFLOW_EXT_LIBRARIES})
