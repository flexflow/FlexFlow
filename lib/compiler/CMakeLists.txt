project(compiler)
set(project_target compiler)

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
  PRIVATE
    src/
  PUBLIC
    include/
)
target_link_libraries(
  ${project_target}
  op-attrs
  utils
  nlohmann_json::nlohmann_json
  optional
  pcg
  spdlog::spdlog
  # substitutions
)
ff_set_cxx_properties(${project_target})
define_ff_vars(${project_target})

add_subdirectory(ffi)
