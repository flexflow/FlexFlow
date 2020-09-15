set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(CUDA_ROOT $ENV{CUDA_ROOT})
list(APPEND CMAKE_PREFIX_PATH ${CUDA_ROOT})
find_package(CUDA REQUIRED)
string(REGEX REPLACE "[^\;]*cudart[^\;]*(\;?)" "" CUDA_LIBRARIES "${CUDA_LIBRARIES}")
set(CUDA_LIBRARIES ${CUDA_LIBRARIES})

# set cuda driver lib
# override cublas and curand because the FindCUDA module may not find the correct libs
if(CUDA_FOUND)
  set(CUDADRV_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libcuda${LIBEXT})
  set(CUDA_CUBLAS_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas${LIBEXT})
  set(CUDA_curand_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcurand${LIBEXT})
endif()

# set CUDA ARCH
# if CUDA_ARCH is not specified, then detect it
if("${CUDA_ARCH}" STREQUAL "")
  include(utils)
  detect_installed_gpus(DETECTED_CUDA_ARCH)
  message( STATUS "CUDA Detected CUDA_ARCH : ${DETECTED_CUDA_ARCH}" )
  set(CUDA_ARCH ${DETECTED_CUDA_ARCH})
endif()
#if CUDA_ARCH is empty
if("${CUDA_ARCH}" STREQUAL "")
  set(CUDA_GENCODE "")
else()
  string(REPLACE "," ";" CUDA_GENCODE "${CUDA_ARCH}")
  string(REGEX REPLACE "([0-9]+)" "-gencode arch=compute_\\1,code=sm_\\1" CUDA_GENCODE "${CUDA_GENCODE}")
endif()

# find cudnn
find_library(CUDNN_LIBRARY 
  NAMES libcudnn${LIBEXT}
  PATHS ${CUDNN_ROOT} ${CUDA_ROOT}
  PATH_SUFFIXES lib lib64
  DOC "CUDNN library." )
  
find_path(CUDNN_INCLUDE_DIR 
    NAMES cudnn.h
    HINTS ${CUDNN_ROOT} ${CUDA_ROOT}
    PATH_SUFFIXES include 
    DOC "CUDNN include directory." )

# find cudnn, set cudnn lib and include    
if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
  set(CUDNN_FOUND ON)
  set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
  set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
endif()

# find cuda and cudnn
if(CUDA_FOUND AND CUDNN_FOUND)
  list(APPEND FLEXFLOW_EXT_LIBRARIES 
    ${CUDADRV_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDNN_LIBRARIES})
endif()

if(CUDA_FOUND)
  message( STATUS "CUDA root path : ${CUDA_TOOLKIT_ROOT_DIR}" )
  message( STATUS "CUDA include path : ${CUDA_INCLUDE_DIRS}" )
  message( STATUS "CUDA runtime libraries : ${CUDA_LIBRARIES}" )
  message( STATUS "CUDA driver libraries : ${CUDADRV_LIBRARIES}" )
  message( STATUS "CUBLAS libraries : ${CUDA_CUBLAS_LIBRARIES}" )
  message( STATUS "CURAND libraries : ${CUDA_curand_LIBRARY}" )
  message( STATUS "CUDA Arch : ${CUDA_GENCODE}" )
else()
  message( FATAL_ERROR "CUDA package not found -> specify search path via CUDA_ROOT variable")
endif()

if(CUDNN_FOUND)
  message( STATUS "CUDNN libraries : ${CUDNN_LIBRARIES}" )
else()
  message( FATAL_ERROR "CUDNN package not found -> specify search path via CUDNN_ROOT variable")
endif()