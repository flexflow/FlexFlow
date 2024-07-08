message(STATUS "TENSORRT Building ...")
set(TENSORRT_NAME tensorrt)

set(TENSORRT_DIR /usr/local/tensorrt)
# Include the TensorRT include directories
include_directories(${TENSORRT_DIR}/include)
# Specify the paths to the TensorRT libraries explicitly
set(TRT_LIB_DIR "/usr/local/tensorrt/lib")
find_library(
  TRT_LIB_PATH nvinfer
  HINTS ${TENSORRT_DIR}/lib
  NO_DEFAULT_PATH)
find_library(TRT_LIB_PATH nvinfer REQUIRED)
message(STATUS "Found nvinfer library: ${TRT_LIB_PATH}")

set(TRTLLM_CPP_INCLUDE_DIR /usr/TensorRT-LLM/cpp/include)
include_directories(/usr/TensorRT-LLM/cpp/include)
set(TENSORRT_LLM_DIR /usr/TensorRT-LLM/tensorrt_llm)
find_library(TRT_LLM_LIB_PATH nvinfer_plugin_tensorrt_llm
             HINTS ${TENSORRT_LLM_DIR}/libs NO_DEFAULT_PATH)
find_library(TRT_LLM_LIB_PATH nvinfer_plugin_tensorrt_llm REQUIRED)
message(STATUS "Found nvinfer_plugin_tensorrt_llm library: ${TRT_LLM_LIB_PATH}")

# Add the library paths
list(APPEND FLEXFLOW_INCLUDE_DIRS ${TENSORRT_DIR}/include)
list(APPEND FLEXFLOW_INCLUDE_DIRS ${TRTLLM_CPP_INCLUDE_DIR})
list(APPEND FLEXFLOW_EXT_LIBRARIES ${TRT_LIB_PATH})
list(APPEND FLEXFLOW_EXT_LIBRARIES ${TRT_LLM_LIB_PATH})
message(STATUS "TENSORRT, TENSORRT-LLM include : ${TRTLLM_CPP_INCLUDE_DIR}, ${TENSORRT_DIR}/include")
message(STATUS "TENSRORT, TENSORRT-LLM libraries : ${TRT_LIB_PATH}, ${TRT_LLM_LIB_PATH}")
add_library(tensorrt SHARED IMPORTED)
