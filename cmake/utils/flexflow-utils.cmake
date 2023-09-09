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

function(ff_get_source_files)
  file(GLOB_RECURSE SRC 
       LIST_DIRECTORIES False
       "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")
  list(FILTER SRC EXCLUDE REGEX "\\.test\\.")
  set(FF_SOURCE_FILES ${SRC} PARENT_SCOPE)
endfunction()

function(ff_add_intree_test_executable)
  ff_parse_args(
    PREFIX 
      FF_TEST_EXEC
    ARGS
      NAME
    VARIADIC_ARGS
      DEPS
    PARSE
      ${ARGN}
  )

  project(${FF_TEST_EXEC_NAME})
  file(GLOB_RECURSE FF_SOURCE_FILES 
       LIST_DIRECTORIES False
       "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")

  add_executable(
    ${FF_TEST_EXEC_NAME}
    ${FF_SOURCE_FILES})

  target_link_libraries(
    ${FF_TEST_EXEC_NAME}
    ${FF_TEST_EXEC_DEPS}
    utils-testing
    utils-rapidcheck_extra
    utils-test_types)

  define_ff_vars(${FF_TEST_EXEC_NAME})
  ff_set_cxx_properties(${FF_TEST_EXEC_NAME})
  doctest_discover_tests(${FF_TEST_EXEC_NAME})
endfunction()

function(ff_add_intree_test_executable)
  ff_parse_args(
    PREFIX 
      FF_TEST_EXEC
    FLAGS
      NO_TEST_LIB
    ARGS
      NAME
    VARIADIC_ARGS
      DEPS
    PARSE
      ${ARGN}
  )

  project(${FF_TEST_EXEC_NAME})
  file(GLOB_RECURSE FF_SOURCE_FILES 
       LIST_DIRECTORIES False
       "${CMAKE_CURRENT_SOURCE_DIR}/src/*.test.cc")

  add_executable(
    ${FF_TEST_EXEC_NAME}
    ${FF_SOURCE_FILES})

  target_link_libraries(
    ${FF_TEST_EXEC_NAME}
    ${FF_TEST_EXEC_DEPS}
    utils-testing)

  define_ff_vars(${FF_TEST_EXEC_NAME})
  ff_set_cxx_properties(${FF_TEST_EXEC_NAME})
  doctest_discover_tests(${FF_TEST_EXEC_NAME})
endfunction()

function(ff_add_library)
  ff_parse_args(
    PREFIX 
      FF_LIBRARY
    FLAGS
      NO_TEST_LIB
    ARGS
      NAME
    VARIADIC_ARGS
      DEPS
      PRIVATE_DEPS
    PARSE
      ${ARGN}
  )
  
  project(${FF_LIBRARY_NAME})
  ff_get_source_files()

  add_library(
    ${FF_LIBRARY_NAME}
    SHARED
    ${FF_SOURCE_FILES})

  target_include_directories(
    ${FF_LIBRARY_NAME}
    PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")

  target_link_libraries(
    ${FF_LIBRARY_NAME}
    PUBLIC
      ${FF_LIBRARY_DEPS}
    PRIVATE
      ${FF_LIBRARY_PRIVATE_DEPS}
  )
  define_ff_vars(${FF_LIBRARY_NAME})
  ff_set_cxx_properties(${FF_LIBRARY_NAME})

  ff_add_intree_test_executable(
    NAME
      "${FF_LIBRARY_NAME}-tests"
    ${FF_LIBRARY_NO_TEST_LIB}
    DEPS
      ${FF_LIBRARY_NAME}
  )
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
  )

  project(${FF_TEST_EXEC_NAME})
  ff_get_source_files()
  file(GLOB_RECURSE IN_TREE_FILES
       LIST_DIRECTORIES False
       "${CMAKE_CURRENT_SOURCE_DIR}/src/*.test.cc")
  list(APPEND FF_SOURCE_FILES ${IN_TREE_FILES})

  add_executable(
    ${FF_TEST_EXEC_NAME}
    ${FF_SOURCE_FILES})

  target_link_libraries(
    ${FF_TEST_EXEC_NAME}
    ${FF_TEST_EXEC_DEPS}
    rapidcheck
    doctest)

  define_ff_vars(${FF_TEST_EXEC_NAME})
  ff_set_cxx_properties(${FF_TEST_EXEC_NAME})
  doctest_discover_tests(${FF_TEST_EXEC_NAME})

endfunction()
