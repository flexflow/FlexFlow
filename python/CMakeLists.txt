cmake_minimum_required(VERSION 3.10)

project(FlexFlow_Python)
set(project_target flexflow_python)

set(CPU_SRC
  flexflow_c.cc
  flexflow_dataloader.cc)

set(GPU_SRC
  flexflow_dataloader.cu)

cuda_add_library(flexflow_dataloader SHARED ${GPU_SRC} ${CPU_SRC} OPTIONS ${CUDA_GENCODE})
target_include_directories(flexflow_dataloader PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
add_dependencies(flexflow_dataloader flexflow)

cuda_add_executable(${project_target} main.cc)
set_target_properties(${project_target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
target_include_directories(${project_target} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(${project_target} -Wl,--whole-archive flexflow -Wl,--no-whole-archive flexflow_dataloader ${FLEXFLOW_EXT_LIBRARIES})

# create pybind bindings
pybind11_add_module(flexflow_pybind11_internal bindings.cc)
set_target_properties(flexflow_pybind11_internal PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/flexflow/core)
target_link_libraries(flexflow_pybind11_internal PRIVATE flexflow flexflow_dataloader ${FLEXFLOW_EXT_LIBRARIES})

# create legion_cffi.py
add_custom_command(TARGET ${project_target} 
  PRE_BUILD	
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/legion_cffi_build.py --runtime-dir ${LEGION_ROOT}/runtime  --defines-dir ${CMAKE_BINARY_DIR}/deps/legion/runtime --output-dir ${CMAKE_CURRENT_SOURCE_DIR}/flexflow/core
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Creating legion_cffi.py..."
)

set(BIN_DEST "bin")
install(TARGETS ${project_target} DESTINATION ${BIN_DEST})
