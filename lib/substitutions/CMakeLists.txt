set(project_target substitutions)
project(${project_target})

file(GLOB_RECURSE SRC
     CONFIGURE_DEPENDS
     LIST_DIRECTORIES False
     src/*.cc)

add_library(
  ${project_target}
  SHARED
  ${SRC}
)
target_include_directories(
  ${project_target}
  PUBLIC
    include/
  PRIVATE
    src/
)
target_link_libraries(
  ${project_target}
  op-attrs
  nlohmann_json::nlohmann_json
)
set_target_properties(
  ${project_target} 
  PROPERTIES 
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

add_subdirectory(ffi)
# add_subdirectory(test)
