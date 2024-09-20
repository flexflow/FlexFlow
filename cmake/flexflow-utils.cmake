macro(ff_parse_args)
  set(flagArgs)
  set(standardArgs PREFIX)
  set(variadicArgs FLAGS ARGS VARIADIC_ARGS PARSE)
  cmake_parse_arguments(FF_PARSE_ARGS "${flagArgs}" "${standardArgs}" "${variadicArgs}" ${ARGN})

  cmake_parse_arguments(${FF_PARSE_ARGS_PREFIX} "${FF_PARSE_ARGS_FLAGS}" "${FF_PARSE_ARGS_ARGS}" "${FF_PARSE_ARGS_VARIADIC_ARGS}" ${FF_PARSE_ARGS_PARSE})
endmacro()

function(define_ff_vars target)
  target_compile_definitions(${target} PRIVATE 
    MAX_OPNAME=${FF_MAX_OPNAME}
    MAX_NUM_OUTPUTS=${FF_MAX_NUM_OUTPUTS}
    MAX_NUM_INPUTS=${FF_MAX_NUM_INPUTS}
    MAX_NUM_WEIGHTS=${FF_MAX_NUM_WEIGHTS}
    MAX_NUM_FUSED_OPERATORS=${FF_MAX_NUM_FUSED_OPERATORS}
    MAX_NUM_FUSED_TENSORS=${FF_MAX_NUM_FUSED_TENSORS}
    MAX_NUM_WORKERS=${FF_MAX_NUM_WORKERS}
    FF_USE_NCCL=${FF_USE_NCCL}
    MAX_TENSOR_DIM=${FF_MAX_DIM}
    MAX_NUM_TASK_REGIONS=${FF_MAX_NUM_TASK_REGIONS}
    MAX_NUM_TASK_ARGUMENTS=${FF_MAX_NUM_TASK_ARGUMENTS}
    )

  if (FF_GPU_BACKEND STREQUAL "cuda")
    target_compile_definitions(${target} PRIVATE FF_USE_CUDA)
  elseif (FF_GPU_BACKEND STREQUAL "hip_cuda")
    target_compile_definitions(${target} PRIVATE FF_USE_HIP_CUDA)
  elseif (FF_GPU_BACKEND STREQUAL "hip_rocm")
    target_compile_definitions(${target} PRIVATE FF_USE_HIP_ROCM)
  endif()
endfunction()

function(ff_set_cxx_properties target)
  set_target_properties(${target}
    PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO
  )
  target_compile_options(${target}
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:> # add C++ compile flags here
  )
endfunction()

function(ff_add_library)
  ff_parse_args(
    PREFIX 
      FF_LIBRARY
    ARGS
      NAME
    VARIADIC_ARGS
      SRC_PATTERNS
      PUBLIC_INCLUDE
      PRIVATE_INCLUDE
      DEPS
      PRIVATE_DEPS
    PARSE
      ${ARGN}
  )
  
  project(${FF_LIBRARY_NAME})
  file(GLOB_RECURSE SRC
       CONFIGURE_DEPENDS 
       LIST_DIRECTORIES False
       ${FF_LIBRARY_SRC_PATTERNS})

  add_library(
    ${FF_LIBRARY_NAME}
    SHARED
    ${SRC})

  target_include_directories(
    ${FF_LIBRARY_NAME}
    PUBLIC
      ${FF_LIBRARY_PUBLIC_INCLUDE}
    PRIVATE
      ${FF_LIBRARY_PRIVATE_INCLUDE})

  target_link_libraries(
    ${FF_LIBRARY_NAME}
    PUBLIC
      ${FF_LIBRARY_DEPS}
    PRIVATE
      ${FF_LIBRARY_PRIVATE_DEPS}
  )
  define_ff_vars(${FF_LIBRARY_NAME})
  ff_set_cxx_properties(${FF_LIBRARY_NAME})
endfunction()

function(ff_add_test_executable)
  ff_parse_args(
    PREFIX 
      FF_TEST_EXEC
    ARGS
      NAME
    VARIADIC_ARGS
      SRC_PATTERNS
      PRIVATE_INCLUDE
      DEPS
    PARSE
      ${ARGN}
      rapidcheck
      doctest
  )

  project(${FF_TEST_EXEC_NAME})
  file(GLOB_RECURSE SRC
       CONFIGURE_DEPENDS
       LIST_DIRECTORIES False
       ${FF_TEST_EXEC_SRC_PATTERNS})

  add_executable(
    ${FF_TEST_EXEC_NAME}
    ${SRC})

  target_link_libraries(
    ${FF_TEST_EXEC_NAME}
    ${FF_TEST_EXEC_DEPS})

  target_compile_definitions(${FF_TEST_EXEC_NAME} PRIVATE FF_TEST_SUITE="${FF_TEST_EXEC_NAME}" FF_CUDA_TEST_SUITE="cuda-${FF_TEST_EXEC_NAME}")

  define_ff_vars(${FF_TEST_EXEC_NAME})
  ff_set_cxx_properties(${FF_TEST_EXEC_NAME})
  doctest_discover_tests(${FF_TEST_EXEC_NAME} ADD_LABELS 1)
endfunction()

function(ff_add_executable)
  ff_parse_args(
    PREFIX 
      FF_EXEC
    ARGS
      NAME
    VARIADIC_ARGS
      SRC_PATTERNS
      PRIVATE_INCLUDE
      DEPS
    PARSE
      ${ARGN}
  )

  project(${FF_EXEC_NAME})
  file(GLOB_RECURSE SRC
       CONFIGURE_DEPENDS
       LIST_DIRECTORIES False
       ${FF_EXEC_SRC_PATTERNS})

  add_executable(
    ${FF_EXEC_NAME}
    ${SRC})

  target_include_directories(
    ${FF_EXEC_NAME}
    PRIVATE
    ${FF_EXEC_PRIVATE_INCLUDE})

  target_link_libraries(
    ${FF_EXEC_NAME}
    ${FF_EXEC_DEPS})

  define_ff_vars(${FF_EXEC_NAME})
  ff_set_cxx_properties(${FF_EXEC_NAME})
endfunction()
