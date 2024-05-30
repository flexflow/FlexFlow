set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(CUDA_ROOT ${CUDA_PATH})
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_PATH})
list(APPEND CMAKE_PREFIX_PATH ${CUDA_ROOT})
find_package(CUDA REQUIRED)

if(CUDA_FOUND)
  # strip the cudart lib
  string(REGEX REPLACE "[^\;]*cudart[^\;]*(\;?)" "" CUDA_LIBRARIES "${CUDA_LIBRARIES}")
  set(CUDA_LIBRARIES ${CUDA_LIBRARIES})

  # set cuda runtime and driver lib
  # override cublas and curand because the FindCUDA module may not find the correct libs  
  set(CUDADRV_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libcuda${LIBEXT})
  if(CUBLAS_PATH)
    set(CUBLAS_ROOT ${CUBLAS_PATH})
  else()
  set(CUBLAS_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
  endif()
  set(CUDA_CUBLAS_LIBRARIES ${CUBLAS_ROOT}/lib64/libcublas${LIBEXT})
  if(CURAND_PATH)
    set(CURAND_ROOT ${CURAND_PATH})
  else()
  set(CURAND_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
  endif()
  set(CUDA_curand_LIBRARY ${CURAND_ROOT}/lib64/libcurand${LIBEXT})
  
  list(APPEND FLEXFLOW_EXT_LIBRARIES
    ${CUDADRV_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY})

  # Snippet below from legion/cmake/newcmake/FindCUDA.cmake
  # Find the `nvcc` executable
  find_program(CUDA_NVCC_EXECUTABLE
    NAMES nvcc
    PATHS "${CUDA_TOOLKIT_ROOT_DIR}"
    ENV CUDA_PATH
    ENV CUDA_BIN_PATH
    PATH_SUFFIXES bin bin64
    NO_DEFAULT_PATH
  )
  # Search default search paths, after we search our own set of paths.
  find_program(CUDA_NVCC_EXECUTABLE nvcc)
  mark_as_advanced(CUDA_NVCC_EXECUTABLE)
  # Compute the CUDA version.
  if(CUDA_NVCC_EXECUTABLE AND NOT CUDA_VERSION)
    execute_process (COMMAND ${CUDA_NVCC_EXECUTABLE} "--version" OUTPUT_VARIABLE NVCC_OUT)
    string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR ${NVCC_OUT})
    string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR ${NVCC_OUT})
    set(CUDA_VERSION "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" CACHE STRING "Version of CUDA as computed from nvcc.")
    mark_as_advanced(CUDA_VERSION)
  else()
    # Need to set these based off of the cached value
    string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR "${CUDA_VERSION}")
    string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR "${CUDA_VERSION}")
  endif()

  # Set FF_CUDA_ARCH to the list of GPU architectures found on the machine.
  if("${FF_CUDA_ARCH}" STREQUAL "autodetect")
    include(utils)
    detect_installed_gpus(DETECTED_CUDA_ARCH)
    message( STATUS "CUDA Detected CUDA_ARCH : ${DETECTED_CUDA_ARCH}" )
    set(FF_CUDA_ARCH ${DETECTED_CUDA_ARCH})
  # Set FF_CUDA_ARCH to the list of all GPU architectures compatible with FlexFlow
  elseif("${FF_CUDA_ARCH}" STREQUAL "all")
    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.8")
      set(FF_CUDA_ARCH 60,61,62,70,72,75,80,86,90)
    else()
      set(FF_CUDA_ARCH 60,61,62,70,72,75,80,86)
    endif()
  endif()
  
  # create CUDA_GENCODE list based on FF_CUDA_ARCH
  string(REPLACE "," ";" CUDA_GENCODE "${FF_CUDA_ARCH}")
  foreach(CODE ${CUDA_GENCODE})
    if(CODE LESS 60)
      message( FATAL_ERROR "CUDA architecture <60 not supported")
    endif()
  endforeach()
  string(REGEX REPLACE "([0-9]+)" "-gencode arch=compute_\\1,code=sm_\\1" CUDA_GENCODE "${CUDA_GENCODE}")

  set(CMAKE_CUDA_COMPILER "${CUDA_NVCC_EXECUTABLE}")
  #output
  message( STATUS "CUDA_VERSION: ${CUDA_VERSION}")
  message( STATUS "CUDA root path : ${CUDA_TOOLKIT_ROOT_DIR}" )
  message( STATUS "CUDA include path : ${CUDA_INCLUDE_DIRS}" )
  message( STATUS "CUDA runtime libraries : ${CUDA_LIBRARIES}" )
  message( STATUS "CUDA driver libraries : ${CUDADRV_LIBRARIES}" )
  message( STATUS "CUBLAS libraries : ${CUDA_CUBLAS_LIBRARIES}" )
  message( STATUS "CURAND libraries : ${CUDA_curand_LIBRARY}" )
  message( STATUS "CUDA Arch : ${FF_CUDA_ARCH}" )
  message( STATUS "CUDA_GENCODE: ${CUDA_GENCODE}")
  message( STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")

  list(APPEND FLEXFLOW_INCLUDE_DIRS
    ${CUDA_INCLUDE_DIRS})

else()
  message( FATAL_ERROR "CUDA package not found -> specify search path via CUDA_ROOT variable")
endif()
