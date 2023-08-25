macro(ff_parse_args)
  set(flagArgs)
  set(standardArgs PREFIX)
  set(variadicArgs FLAGS ARGS VARIADIC_ARGS PARSE)
  cmake_parse_arguments(FF_PARSE_ARGS "${flagArgs}" "${standardArgs}" "${variadicArgs}" ${ARGN})

  cmake_parse_arguments(${FF_PARSE_ARGS_PREFIX} "${FF_PARSE_ARGS_FLAGS}" "${FF_PARSE_ARGS_ARGS}" "${FF_PARSE_ARGS_VARIADIC_ARGS}" ${FF_PARSE_ARGS_PARSE})
endmacro()

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

  define_ff_vars(${FF_TEST_EXEC_NAME})
  ff_set_cxx_properties(${FF_TEST_EXEC_NAME})
  doctest_discover_tests(${FF_TEST_EXEC_NAME})
endfunction()
