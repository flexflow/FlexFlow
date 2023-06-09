add_library(
  visit_struct
  INTERFACE
)
target_include_directories(
  visit_struct
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/visit_struct/include/
)
set_target_properties(
  ${project_target} 
  PROPERTIES 
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
