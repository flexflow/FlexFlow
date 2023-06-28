add_library(
  any
  INTERFACE
)
target_include_directories(
  any
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/any/
)
set_target_properties(
  any
  PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
