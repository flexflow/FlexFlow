# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/variant)

# list(APPEND FLEXFLOW_EXT_LIBRARIES mpark_variant)
# list(APPEND FLEXFLOW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/deps/variant/include/)

FetchContent_Declare(variant GIT_REPOSITORY https://github.com/mpark/variant)
FetchContent_MakeAvailable(variant)
#list(APPEND FLEXFLOW_EXT_LIBRARIES mpark_variant)
list(APPEND FLEXFLOW_INCLUDE_DIRS ${variant_SOURCE_DIR}/include/)
