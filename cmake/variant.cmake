FetchContent_Declare(variant GIT_REPOSITORY https://github.com/mpark/variant)
FetchContent_MakeAvailable(variant)
list(APPEND FLEXFLOW_INCLUDE_DIRS ${variant_SOURCE_DIR}/include/)
