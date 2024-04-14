include(aliasing)
  
if (FF_USE_EXTERNAL_DOCTEST)
  find_package(doctest REQUIRED)
  include(doctest) # import doctest_discover_tests
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/doctest)
  include(${CMAKE_CURRENT_SOURCE_DIR}/deps/doctest/scripts/cmake/doctest.cmake)
endif()

alias_library(doctest doctest::doctest)
