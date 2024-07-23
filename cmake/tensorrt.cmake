message(STATUS "TENSORRT Building ...")
set(TENSORRT_NAME tensorrt)

set(TENSORRT_DIR /usr/local/tensorrt)
# Include the TensorRT include directories
include_directories(${TENSORRT_DIR}/include)
# Specify the paths to the TensorRT libraries explicitly
find_library(
  TENSORRT_LIB_PATH nvinfer
  HINTS ${TENSORRT_DIR}/lib
  NO_DEFAULT_PATH)
# find_library(TENSORRT_LIB_PATH nvinfer REQUIRED)
message(STATUS "Found nvinfer library: ${TENSORRT_LIB_PATH}")

add_library(tensorrt SHARED IMPORTED)
set_target_properties(tensorrt PROPERTIES IMPORTED_LOCATION ${TENSORRT_DIR})

list(APPEND FLEXFLOW_INCLUDE_DIRS ${TENSORRT_DIR}/include)
list(APPEND FLEXFLOW_EXT_LIBRARIES ${TENSORRT_DIR}/lib/libnvinfer.so)
