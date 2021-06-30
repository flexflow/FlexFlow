# find cudnn in CUDNN_ROOT and CUDA_ROOT
if(CUDNN_PATH)
  set(CUDNN_ROOT ${CUDNN_PATH})
else()
	# if CUDNN_PATH is not set, let's try to find it in the CUDA root
	set(CUDNN_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
endif()
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
if(CUDNN_FOUND)
  list(APPEND FLEXFLOW_EXT_LIBRARIES
    ${CUDNN_LIBRARIES})

  list(APPEND FLEXFLOW_INCLUDE_DIRS
    ${CUDNN_INCLUDE_DIR})
endif()

if(CUDNN_FOUND)
message( STATUS "CUDNN inlcude : ${CUDNN_INCLUDE_DIR}" )
  message( STATUS "CUDNN libraries : ${CUDNN_LIBRARIES}" )
else()
  message( FATAL_ERROR "CUDNN package not found -> specify search path via CUDNN_DIR variable")
endif()
