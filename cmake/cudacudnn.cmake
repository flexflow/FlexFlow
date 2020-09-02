set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)
string(REGEX REPLACE "[^\;]*cudart[^\;]*(\;?)" "" CUDA_LIBRARIES "${CUDA_LIBRARIES}")
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} PARENT_SCOPE)

# set cuda driver lib
# override cublas and curand because the FindCUDA module may not find the correct libs
set(CUDADRV_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libcuda${LIBEXT})
set(CUDA_CUBLAS_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas${LIBEXT})
set(CUDA_curand_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcurand${LIBEXT})

# set cudnn lib
set(CUDNN_INCLUDE_DIRS /scratch/shared/wwu12/cudnn/include)
set(CUDNN_LIBRARIES /scratch/shared/wwu12/cudnn/lib64/libcudnn${LIBEXT})

list(APPEND FLEXFLOW_EXT_LIBRARIES 
  ${CUDADRV_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDNN_LIBRARIES})

if(CUDA_FOUND)
  set(CUDA_CUDNN_FOUND ON)
endif()

if(CUDA_CUDNN_FOUND)
  message( STATUS "CUDA root path : ${CUDA_TOOLKIT_ROOT_DIR}" )
  message( STATUS "CUDA include path : ${CUDA_INCLUDE_DIRS}" )
  message( STATUS "CUDA driver libraries : ${CUDADRV_LIBRARIES}" )
  message( STATUS "CUBLAS libraries : ${CUDA_CUBLAS_LIBRARIES}" )
  message( STATUS "CURAND libraries : ${CUDA_curand_LIBRARY}" )
  message( STATUS "CUDNN libraries : ${CUDNN_LIBRARIES}" )
else()
  message( WARNING "CUDA package not found -> specify search path via PROTOBUF_ROOT variable")
endif()