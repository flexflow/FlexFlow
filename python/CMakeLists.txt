cmake_minimum_required(VERSION 3.10)

project(FlexFlow_Python)
set(project_target flexflow_python)

set(CPU_SRC
  flexflow_c.cc
  flexflow_dataloader.cc)

set(GPU_SRC
  flexflow_dataloader.cu)

cuda_add_library(flexflow_c SHARED ${GPU_SRC} ${CPU_SRC} OPTIONS ${CUDA_GENCODE})
target_include_directories(flexflow_c PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
add_dependencies(flexflow_c flexflow)

cuda_add_executable(${project_target} main.cc)
set_target_properties(${project_target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
target_include_directories(${project_target} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(${project_target} -Wl,--whole-archive flexflow -Wl,--no-whole-archive flexflow_c ${FLEXFLOW_EXT_LIBRARIES})

# create pybind bindings
pybind11_add_module(flexflow_pybind11_internal bindings.cc)
set_target_properties(flexflow_pybind11_internal PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/flexflow/core)
target_link_libraries(flexflow_pybind11_internal PRIVATE flexflow flexflow_c ${FLEXFLOW_EXT_LIBRARIES})

# create legion_cffi_header.py
add_custom_command(TARGET ${project_target} 
  PRE_BUILD	
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/legion_cffi_build.py --runtime-dir ${LEGION_ROOT}/runtime  --defines-dir ${CMAKE_BINARY_DIR}/deps/legion/runtime --output-dir ${CMAKE_CURRENT_SOURCE_DIR}/flexflow/core
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Creating legion_cffi_header.py..."
)

# create flexflow_cffi_header.py
add_custom_command(TARGET ${project_target} 
  PRE_BUILD	
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/flexflow_cffi_build.py --ffhome-dir ${FLEXFLOW_ROOT} --output-dir ${CMAKE_CURRENT_SOURCE_DIR}/flexflow/core
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Creating flexflow_cffi_header.py..."
)

set(BIN_DEST "bin")
set(LIB_DEST "lib")
install(TARGETS ${project_target} DESTINATION ${BIN_DEST})
install(TARGETS flexflow_c DESTINATION ${LIB_DEST})
if(FF_BUILD_FROM_PYPI)
  set(PY_DEST "./")
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/flexflow/core/legion_cffi_header.py ${CMAKE_CURRENT_SOURCE_DIR}/flexflow/core/flexflow_cffi_header.py 
	  DESTINATION ${PY_DEST}/core)
else()
	set(PY_DEST "flexflow")
	install(
		DIRECTORY ${PROJECT_SOURCE_DIR}/flexflow/ 
		DESTINATION ${PY_DEST}
		FILES_MATCHING 
		PATTERN "*.py")
endif()
 install(TARGETS flexflow_pybind11_internal DESTINATION ${PY_DEST}/core)