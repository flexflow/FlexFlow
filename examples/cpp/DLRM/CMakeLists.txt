cmake_minimum_required(VERSION 3.10)

project(FlexFlowExample_DLRM)
set(project_target dlrm)
find_package(HDF5 REQUIRED COMPONENTS C)
set(CPU_SRC
  ${FLEXFLOW_CPP_DRV_SRC}
  dlrm.cc
  dlrm.h)

set(GPU_SRC
  dlrm.cu)

cuda_add_executable(${project_target} ${CPU_SRC} ${GPU_SRC})
target_include_directories(${project_target} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR} ${HDF5_C_INCLUDE_DIRS})
target_link_libraries(${project_target} -Wl,--whole-archive flexflow -Wl,--no-whole-archive ${FLEXFLOW_EXT_LIBRARIES} ${HDF5_C_LIBRARIES})

set(BIN_DEST "bin")
install(TARGETS ${project_target} DESTINATION ${BIN_DEST})
