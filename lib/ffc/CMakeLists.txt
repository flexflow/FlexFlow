project(ffc)
set(project_target ffc)

file(GLOB_RECURSE ffc_SRC
     CONFIGURE_DEPENDS
     LIST_DIRECTORIES False
     src/substitution_loader.cc)

add_library(
  ${project_target}
  SHARED
  ${ffc_SRC}
)
target_include_directories(
  ${project_target}
  PRIVATE
    src/
  PUBLIC
    include/
)
target_link_libraries(
  ${project_target}
  op-meta
  utils
  nlohmann_json::nlohmann_json
  optional
  pcg
  spdlog::spdlog
)
set_target_properties(
  ${project_target} 
  PROPERTIES 
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

list(APPEND FLEXFLOW_LIBS ${PROJECT_SOURCE_DIR}/lib${project_target}${LIBEXT})
list(APPEND FLEXFLOW_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)

set(FLEXFLOW_LIBS ${FLEXFLOW_LIBS} PARENT_SCOPE)
set(FLEXFLOW_INCLUDE_DIRS ${FLEXFLOW_INCLUDE_DIRS} PARENT_SCOPE)
