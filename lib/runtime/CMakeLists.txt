set(project_target ffr)

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
  PRIVATE
    src/
  INTERFACE
    include/
)
target_link_libraries(
  ${project_target}
  op-meta
  utils
  optional
  legion
)

