cmake_minimum_required(VERSION 3.10)

project(FlexFlow_Python)
set(project_target flexflow_python)

set(CPU_SRC
  flexflow_c.cc
  flexflow_dataloader.cc)

if (FF_GPU_BACKEND STREQUAL "cuda")
  set(GPU_SRC
    flexflow_dataloader.cu)

  cuda_add_library(flexflow_c SHARED ${GPU_SRC} ${CPU_SRC} OPTIONS ${CUDA_GENCODE})
  cuda_add_library(flexflow_native_python SHARED ${GPU_SRC} ${CPU_SRC} OPTIONS ${CUDA_GENCODE})
elseif (FF_GPU_BACKEND STREQUAL "hip_cuda" OR FF_GPU_BACKEND STREQUAL "hip_rocm")
  set(GPU_SRC
    flexflow_dataloader.cpp)

  add_library(flexflow_c SHARED ${GPU_SRC} ${CPU_SRC})
  add_library(flexflow_native_python SHARED ${GPU_SRC} ${CPU_SRC})
else()
  message(FATAL_ERROR "Unsupported FF_GPU_BACKEND for cmake: ${FF_GPU_BACKEND}")
endif()

target_include_directories(flexflow_c PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(flexflow_c ${LEGION_LIBRARY})
add_dependencies(flexflow_c flexflow)

target_include_directories(flexflow_native_python PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(flexflow_native_python flexflow ${LEGION_LIBRARY} ${FLEXFLOW_EXT_LIBRARIES})
add_dependencies(flexflow_native_python flexflow)

if (FF_GPU_BACKEND STREQUAL "cuda")
  cuda_add_executable(${project_target} main.cc)
elseif (FF_GPU_BACKEND STREQUAL "hip_cuda" OR FF_GPU_BACKEND STREQUAL "hip_rocm")
  add_executable(${project_target} main.cc)
else()
  message(FATAL_ERROR "Unsupported FF_GPU_BACKEND for cmake: ${FF_GPU_BACKEND}")
endif()

set_target_properties(${project_target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
target_include_directories(${project_target} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(${project_target} -Wl,--whole-archive flexflow -Wl,--no-whole-archive flexflow_c ${FLEXFLOW_EXT_LIBRARIES})

# create legion_cffi_header.py
if(FF_USE_EXTERNAL_LEGION)
  set(LEGION_HEADER_DIR ${LEGION_ROOT}/include)
  set(LEGION_DEFINE_DIR ${LEGION_ROOT}/include)
else()
  set(LEGION_HEADER_DIR ${LEGION_INCLUDE_DIR})
  set(LEGION_DEFINE_DIR ${LEGION_DEF_DIR})
endif()
add_custom_command(TARGET ${project_target} 
  PRE_BUILD	
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/legion_cffi_build.py --runtime-dir ${LEGION_HEADER_DIR}  --defines-dir ${LEGION_DEFINE_DIR} --output-dir ${CMAKE_CURRENT_SOURCE_DIR}/flexflow/core
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Creating legion_cffi_header.py..."
)

# create flexflow_cffi_header.py
add_custom_command(TARGET ${project_target} 
  PRE_BUILD	
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/flexflow_cffi_build.py --ffhome-dir ${FLEXFLOW_ROOT} --libname libflexflow_native_python.so --output-dir ${CMAKE_CURRENT_SOURCE_DIR}/flexflow/core
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Creating flexflow_cffi_header.py..."
)

set(BIN_DEST "bin")
set(LIB_DEST "lib")
install(TARGETS ${project_target} DESTINATION ${BIN_DEST})
install(TARGETS flexflow_c DESTINATION ${LIB_DEST})
install(TARGETS flexflow_native_python DESTINATION ${LIB_DEST})
if(FF_BUILD_FROM_PYPI)
  set(PY_DEST "./")
	install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/flexflow/core/legion_cffi_header.py ${CMAKE_CURRENT_SOURCE_DIR}/flexflow/core/flexflow_cffi_header.py 
	  DESTINATION ${PY_DEST}/core)
else()
  execute_process(COMMAND python -c "from distutils import sysconfig; print(sysconfig.get_python_lib(plat_specific=False,standard_lib=False))" OUTPUT_VARIABLE PY_DEST OUTPUT_STRIP_TRAILING_WHITESPACE)
	install(
		DIRECTORY ${PROJECT_SOURCE_DIR}/flexflow/ 
		DESTINATION ${PY_DEST}/flexflow
		FILES_MATCHING 
		PATTERN "*.py")
endif()
