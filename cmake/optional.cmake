set(OPTIONAL_BUILD_TESTS OFF)
set(OPTIONAL_BUILD_PACKAGE OFF)

FetchContent_Declare(optional GIT_REPOSITORY https://github.com/TartanLlama/optional.git)
FetchContent_MakeAvailable(optional)
list(APPEND FLEXFLOW_INCLUDE_DIRS ${optional_SOURCE_DIR}/include/)
