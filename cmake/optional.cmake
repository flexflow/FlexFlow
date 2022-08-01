add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/optional)

list(APPEND FLEXFLOW_EXT_LIBRARIES optional)
list(APPEND FLEXFLOW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/deps/optional/include/)
