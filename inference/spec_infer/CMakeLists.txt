cmake_minimum_required(VERSION 3.10)

project(FlexFlow_SpecInfer)
set(project_target spec_infer)


set(CPU_SRC
  ${FLEXFLOW_CPP_DRV_SRC}
  spec_infer.cc
  ../models/llama.cc
  ../models/opt.cc
  ../models/falcon.cc
  ../models/mpt.cc)

if (FF_GPU_BACKEND STREQUAL "cuda" OR FF_GPU_BACKEND STREQUAL "hip_cuda")
  cuda_add_executable(${project_target} ${CPU_SRC})
  if (FF_GPU_BACKEND STREQUAL "hip_cuda")
    target_compile_definitions(${project_target} PRIVATE __HIP_PLATFORM_NVIDIA__)
  endif()
elseif(FF_GPU_BACKEND STREQUAL "hip_rocm")
  set_source_files_properties(${CPU_SRC} PROPERTIES LANGUAGE HIP)
  hip_add_executable(${project_target} ${CPU_SRC})
  if (FF_HIP_ARCH STREQUAL "")
    message(FATAL_ERROR "FF_HIP_ARCH is empty!")
  endif()
  set_property(TARGET ${project_target} PROPERTY HIP_ARCHITECTURES "${FF_HIP_ARCH}")
  target_compile_definitions(${project_target} PRIVATE __HIP_PLATFORM_AMD__)
else()
  message(FATAL_ERROR "Compilation of ${project_target} for ${FF_GPU_BACKEND} backend not yet supported")
endif()

target_include_directories(${project_target} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_include_directories(${project_target} PRIVATE ${CMAKE_SOURCE_DIR}/inference)
target_link_libraries(${project_target} -Wl,--whole-archive flexflow -Wl,--no-whole-archive ${FLEXFLOW_EXT_LIBRARIES})

set(BIN_DEST "bin")
install(TARGETS ${project_target} DESTINATION ${BIN_DEST})
