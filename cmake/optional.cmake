set(OPTIONAL_BUILD_TESTS OFF)
set(OPTIONAL_BUILD_PACKAGE OFF)

# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/optional)

# list(APPEND FLEXFLOW_EXT_LIBRARIES optional)
# list(APPEND FLEXFLOW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/deps/optional/include/)
FetchContent_Declare(optional GIT_REPOSITORY https://github.com/TartanLlama/optional.git)
FetchContent_MakeAvailable(optional)
#list(APPEND FLEXFLOW_EXT_LIBRARIES optional)
list(APPEND FLEXFLOW_INCLUDE_DIRS ${optional_SOURCE_DIR}/include/)
