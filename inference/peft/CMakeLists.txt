cmake_minimum_required(VERSION 3.10)

project(FlexFlow_Peft)

# Normal PEFT
set(project_target1 peft)
set(CPU_SRC1
  ${FLEXFLOW_CPP_DRV_SRC}
  peft.cc
  ../models/llama.cc
  ../models/opt.cc
  ../models/falcon.cc
  ../models/starcoder.cc
  ../models/mpt.cc)

if (FF_GPU_BACKEND STREQUAL "cuda" OR FF_GPU_BACKEND STREQUAL "hip_cuda")
  cuda_add_executable(${project_target1} ${CPU_SRC1})
  if (FF_GPU_BACKEND STREQUAL "hip_cuda")
    target_compile_definitions(${project_target1} PRIVATE __HIP_PLATFORM_NVIDIA__)
  endif()
elseif(FF_GPU_BACKEND STREQUAL "hip_rocm")
  set_source_files_properties(${CPU_SRC1} PROPERTIES LANGUAGE HIP)
  hip_add_executable(${project_target1} ${CPU_SRC1})
  if (FF_HIP_ARCH STREQUAL "")
    message(FATAL_ERROR "FF_HIP_ARCH is empty!")
  endif()
  set_property(TARGET ${project_target1} PROPERTY HIP_ARCHITECTURES "${FF_HIP_ARCH}")
  target_compile_definitions(${project_target1} PRIVATE __HIP_PLATFORM_AMD__)
else()
  message(FATAL_ERROR "Compilation of ${project_target1} for ${FF_GPU_BACKEND} backend not yet supported")
endif()

target_include_directories(${project_target1} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_include_directories(${project_target1} PRIVATE ${CMAKE_SOURCE_DIR}/inference)
target_link_libraries(${project_target1} -Wl,--whole-archive flexflow -Wl,--no-whole-archive ${FLEXFLOW_EXT_LIBRARIES})
set(BIN_DEST "bin")
install(TARGETS ${project_target1} DESTINATION ${BIN_DEST})

# FWD benchmark
set(project_target2 peft_fwd_benchmark)
set(CPU_SRC2
  ${FLEXFLOW_CPP_DRV_SRC}
  peft_fwd_benchmark.cc
  ../models/llama.cc
  ../models/opt.cc
  ../models/falcon.cc
  ../models/starcoder.cc
  ../models/mpt.cc)

if (FF_GPU_BACKEND STREQUAL "cuda" OR FF_GPU_BACKEND STREQUAL "hip_cuda")
  cuda_add_executable(${project_target2} ${CPU_SRC2})
  if (FF_GPU_BACKEND STREQUAL "hip_cuda")
    target_compile_definitions(${project_target2} PRIVATE __HIP_PLATFORM_NVIDIA__)
  endif()
elseif(FF_GPU_BACKEND STREQUAL "hip_rocm")
  set_source_files_properties(${CPU_SRC2} PROPERTIES LANGUAGE HIP)
  hip_add_executable(${project_target2} ${CPU_SRC2})
  if (FF_HIP_ARCH STREQUAL "")
    message(FATAL_ERROR "FF_HIP_ARCH is empty!")
  endif()
  set_property(TARGET ${project_target2} PROPERTY HIP_ARCHITECTURES "${FF_HIP_ARCH}")
  target_compile_definitions(${project_target2} PRIVATE __HIP_PLATFORM_AMD__)
else()
  message(FATAL_ERROR "Compilation of ${project_target2} for ${FF_GPU_BACKEND} backend not yet supported")
endif()

target_include_directories(${project_target2} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_include_directories(${project_target2} PRIVATE ${CMAKE_SOURCE_DIR}/inference)
target_link_libraries(${project_target2} -Wl,--whole-archive flexflow -Wl,--no-whole-archive ${FLEXFLOW_EXT_LIBRARIES})
set(BIN_DEST "bin")
install(TARGETS ${project_target2} DESTINATION ${BIN_DEST})

# BWD benchmark
set(project_target3 peft_bwd_benchmark)
set(CPU_SRC3
  ${FLEXFLOW_CPP_DRV_SRC}
  peft_bwd_benchmark.cc
  ../models/llama.cc
  ../models/opt.cc
  ../models/falcon.cc
  ../models/starcoder.cc
  ../models/mpt.cc)

if (FF_GPU_BACKEND STREQUAL "cuda" OR FF_GPU_BACKEND STREQUAL "hip_cuda")
  cuda_add_executable(${project_target3} ${CPU_SRC3})
  if (FF_GPU_BACKEND STREQUAL "hip_cuda")
    target_compile_definitions(${project_target3} PRIVATE __HIP_PLATFORM_NVIDIA__)
  endif()
elseif(FF_GPU_BACKEND STREQUAL "hip_rocm")
  set_source_files_properties(${CPU_SRC3} PROPERTIES LANGUAGE HIP)
  hip_add_executable(${project_target3} ${CPU_SRC3})
  if (FF_HIP_ARCH STREQUAL "")
    message(FATAL_ERROR "FF_HIP_ARCH is empty!")
  endif()
  set_property(TARGET ${project_target3} PROPERTY HIP_ARCHITECTURES "${FF_HIP_ARCH}")
  target_compile_definitions(${project_target3} PRIVATE __HIP_PLATFORM_AMD__)
else()
  message(FATAL_ERROR "Compilation of ${project_target3} for ${FF_GPU_BACKEND} backend not yet supported")
endif()

target_include_directories(${project_target3} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_include_directories(${project_target3} PRIVATE ${CMAKE_SOURCE_DIR}/inference)
target_link_libraries(${project_target3} -Wl,--whole-archive flexflow -Wl,--no-whole-archive ${FLEXFLOW_EXT_LIBRARIES})
set(BIN_DEST "bin")
install(TARGETS ${project_target3} DESTINATION ${BIN_DEST})

# Online peft
set(project_target4 req_rate_benchmark)
set(CPU_SRC4
  ${FLEXFLOW_CPP_DRV_SRC}
  req_rate_benchmark.cc
  ../models/llama.cc
  ../models/opt.cc
  ../models/falcon.cc
  ../models/starcoder.cc
  ../models/mpt.cc)

if (FF_GPU_BACKEND STREQUAL "cuda" OR FF_GPU_BACKEND STREQUAL "hip_cuda")
  cuda_add_executable(${project_target4} ${CPU_SRC4})
  if (FF_GPU_BACKEND STREQUAL "hip_cuda")
    target_compile_definitions(${project_target4} PRIVATE __HIP_PLATFORM_NVIDIA__)
  endif()
elseif(FF_GPU_BACKEND STREQUAL "hip_rocm")
  set_source_files_properties(${CPU_SRC4} PROPERTIES LANGUAGE HIP)
  hip_add_executable(${project_target4} ${CPU_SRC4})
  if (FF_HIP_ARCH STREQUAL "")
    message(FATAL_ERROR "FF_HIP_ARCH is empty!")
  endif()
  set_property(TARGET ${project_target4} PROPERTY HIP_ARCHITECTURES "${FF_HIP_ARCH}")
  target_compile_definitions(${project_target4} PRIVATE __HIP_PLATFORM_AMD__)
else()
  message(FATAL_ERROR "Compilation of ${project_target4} for ${FF_GPU_BACKEND} backend not yet supported")
endif()

target_include_directories(${project_target4} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_include_directories(${project_target4} PRIVATE ${CMAKE_SOURCE_DIR}/inference)
target_link_libraries(${project_target4} -Wl,--whole-archive flexflow -Wl,--no-whole-archive ${FLEXFLOW_EXT_LIBRARIES})
set(BIN_DEST "bin")
install(TARGETS ${project_target4} DESTINATION ${BIN_DEST})
