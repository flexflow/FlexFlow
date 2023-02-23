set(project_target utils)
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
  optional
  mpark_variant
)
set_target_properties(
  ${project_target} 
  PROPERTIES 
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# add_subdirectory(test)

list(APPEND FLEXFLOW_LIBS ${PROJECT_SOURCE_DIR}/lib${project_target}${LIBEXT})
list(APPEND FLEXFLOW_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)

set(FLEXFLOW_LIBS ${FLEXFLOW_LIBS} PARENT_SCOPE)
set(FLEXFLOW_INCLUDE_DIRS ${FLEXFLOW_INCLUDE_DIRS} PARENT_SCOPE)
