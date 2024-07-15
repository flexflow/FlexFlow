message(STATUS "TENSORRT Building ...")
set(TENSORRT_NAME tensorrt_llm)

set(TENSORRT_DIR /usr/local/tensorrt)
# Include the TensorRT include directories
include_directories(${TENSORRT_DIR}/include)
# Specify the paths to the TensorRT libraries explicitly
# set(TRT_LIB_DIR /usr/local/tensorrt/lib)
find_library(
  TENSORRT_LIB_PATH nvinfer
  HINTS ${TENSORRT_DIR}/lib
  NO_DEFAULT_PATH)
# find_library(TENSORRT_LIB_PATH nvinfer REQUIRED)
message(STATUS "Found nvinfer library: ${TENSORRT_LIB_PATH}")

add_library(tensorrt SHARED IMPORTED)
set_target_properties(tensorrt PROPERTIES IMPORTED_LOCATION ${TENSORRT_DIR})


set(TENSORRT_LLM_CPP_DIR /usr/TensorRT-LLM/cpp)
include_directories(${TENSORRT_LLM_CPP_DIR}/include)
set(TENSORRT_LLM_DIR /usr/TensorRT-LLM/tensorrt_llm)
find_library(TENSORRT_LLM_LIB_PATH nvinfer_plugin_tensorrt_llm
             HINTS ${TENSORRT_LLM_DIR}/libs NO_DEFAULT_PATH)
# find_library(TENSORRT_LLM_LIB_PATH nvinfer_plugin_tensorrt_llm REQUIRED)
message(STATUS "Found nvinfer_plugin_tensorrt_llm library: ${TENSORRT_LLM_LIB_PATH}")

# Add the library paths
add_library(tensorrt_llm SHARED IMPORTED)
set_target_properties(tensorrt_llm PROPERTIES IMPORTED_LOCATION ${TENSORRT_LLM_DIR})

list(APPEND FLEXFLOW_INCLUDE_DIRS ${TENSORRT_DIR}/include)
list(APPEND FLEXFLOW_INCLUDE_DIRS ${TENSORRT_LLM_CPP_DIR}/include)
# list(APPEND FLEXFLOW_INCLUDE_DIRS /usr/TensorRT-LLM/cpp)

list(APPEND FLEXFLOW_EXT_LIBRARIES ${TENSORRT_DIR}/lib/libnvinfer.so)
list(APPEND FLEXFLOW_EXT_LIBRARIES ${TENSORRT_LLM_DIR}/libs/libtensorrt_llm.so)
list(APPEND FLEXFLOW_EXT_LIBRARIES ${TENSORRT_LLM_DIR}/libs/libnvinfer_plugin_tensorrt_llm.so)
list(APPEND FLEXFLOW_EXT_LIBRARIES ${TENSORRT_LLM_DIR}/libs/libtensorrt_llm_nvrtc_wrapper.so)
list(APPEND FLEXFLOW_EXT_LIBRARIES ${TENSORRT_LLM_DIR}/libs/libth_common.so)
# message(STATUS "TENSORRT, TENSORRT-LLM include : ${TRTLLM_CPP_INCLUDE_DIR}, ${TENSORRT_DIR}/include")
# message(STATUS "TENSRORT, TENSORRT-LLM libraries : ${TRT_LIB_PATH}, ${TRT_LLM_LIB_PATH}")
# add_library(tensorrt SHARED IMPORTED)

