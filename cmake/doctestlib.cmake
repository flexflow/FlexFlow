include(aliasing)
  
if (FF_USE_EXTERNAL_DOCTEST)
  find_package(doctest REQUIRED)
  include(doctest) # import doctest_discover_tests
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/doctest)
  include(${CMAKE_CURRENT_SOURCE_DIR}/deps/doctest/scripts/cmake/doctest.cmake)
endif()

target_compile_definitions(
  doctest::doctest
  INTERFACE
    DOCTEST_CONFIG_REQUIRE_STRINGIFICATION_FOR_ALL_USED_TYPES
)
alias_library(doctest doctest::doctest)
