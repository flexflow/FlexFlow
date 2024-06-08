message(STATUS "Building TensorRT-LLM from source")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/tensorrt-llm)
list(APPEND FLEXFLOW_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/deps/tensorrt-llm/cpp/tensorrt_llm/include/)
set(TENSORRT_LIBRARY Tensorrt)
set(TENSORRT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/deps/tensorrt-llm/cpp/tensorrt_llm)
set(TENSORRT_DEF_DIR ${CMAKE_BINARY_DIR}/deps/tensorrt-llm/cpp/tensorrt_llm)

# Add TensorRT to the FlexFlow build
if(TENSORRT_LIBRARY)
    list(APPEND FF_CC_FLAGS -DUSE_TENSORRT)
    list(APPEND FF_NVCC_FLAGS -DUSE_TENSORRT)
    target_include_directories(flexflow PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(flexflow PRIVATE ${TENSORRT_LIBRARY})
endif()

if (FF_GPU_BACKEND STREQUAL "cuda")
	set(TensorRT_USE_CUDA ON CACHE BOOL "enable TensorRT_USE_CUDA" FORCE)
	set(TensorRT_CUDA_ARCH ${FF_CUDA_ARCH} CACHE STRING "TensorRT CUDA ARCH" FORCE)
endif()
