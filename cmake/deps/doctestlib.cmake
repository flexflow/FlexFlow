include(aliasing)
  
if (FF_USE_EXTERNAL_DOCTEST)
  find_package(doctest REQUIRED)
  include(doctest) # import doctest_discover_tests
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/doctest)
  include(${CMAKE_CURRENT_SOURCE_DIR}/deps/doctest/scripts/cmake/doctest.cmake)
endif()

add_library(doctest-ff INTERFACE)
target_compile_definitions(doctest-ff INTERFACE DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS)
target_link_libraries(doctest-ff INTERFACE doctest::doctest)
alias_library(doctest doctest-ff)
