set(NCCL_NAME nccl)
# set(NCCL_CUDA_ARCH "-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
# message("NCCL_CUDA_ARCH: ${NCCL_CUDA_ARCH}")

if(NCCL_PATH)
  set(NCCL_ROOT ${NCCL_PATH})
else()
  # if NCCL_PATH is not set, let's try to find it in the CUDA root
  set(NCCL_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
endif()

find_library(NCCL_LIBRARY
  NAMES libnccl${LIBEXT}
  PATHS ${NCCL_ROOT} ${CUDA_ROOT}
  PATH_SUFFIXES lib lib64
  DOC "NCCL library." )

find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS ${NCCL_ROOT}
  PATH_SUFFIXES include 
  DOC "NCCL include directory.")

# find NCCL, set NCCL lib and include    
if(NCCL_LIBRARY AND NCCL_INCLUDE_DIR)
  set(NCCL_FOUND ON)
  set(NCCL_LIBRARIES ${NCCL_LIBRARY})
  set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})

  # Check NCCL version
  if(EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
    file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" NCCL_VERSION_DEFINES
         REGEX "#define NCCL_MAJOR [0-9]+" )
    file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" NCCL_VERSION_DEFINES2
         REGEX "#define NCCL_MINOR [0-9]+" )
    string(REGEX MATCH "([0-9]+)" NCCL_MAJOR ${NCCL_VERSION_DEFINES})
    string(REGEX MATCH "([0-9]+)" NCCL_MINOR ${NCCL_VERSION_DEFINES2})
    set(NCCL_VERSION "${NCCL_MAJOR}.${NCCL_MINOR}")
    set(NCCL_OLD FALSE)
    # if(NCCL_VERSION VERSION_LESS 2.23)
    #   set(NCCL_OLD TRUE)
    # else()
    #   set(NCCL_OLD FALSE)
    # endif()
    message(STATUS "Found NCCL version: ${NCCL_VERSION}")
  else()
    message(WARNING "NCCL header not found, unable to determine version")
    set(NCCL_OLD TRUE)  # Assume old version if we can't determine
  endif()
endif()

# find NCCL
if(NCCL_FOUND AND (NOT NCCL_OLD OR CUDA_VERSION VERSION_LESS 12.0))
  list(APPEND FLEXFLOW_EXT_LIBRARIES ${NCCL_LIBRARIES})
  list(APPEND FLEXFLOW_INCLUDE_DIRS ${NCCL_INCLUDE_DIRS})
  message( STATUS "NCCL include : ${NCCL_INCLUDE_DIRS}" )
  message( STATUS "NCCL libraries : ${NCCL_LIBRARIES}" )
  add_library(nccl SHARED IMPORTED)

# Build NCCL from source
else()
  message(STATUS "Building NCCL from source")
  list(TRANSFORM CUDA_GENCODE PREPEND "NVCC_GENCODE=" OUTPUT_VARIABLE NCCL_BUILD_NVCC_GENCODE)

  set(NCCL_BUILD_CMD make src.build "${NCCL_BUILD_NVCC_GENCODE}" "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}" "BUILDDIR=${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}")
  if(DEFINED ENV{MAKEFLAGS})
    set(NCCL_BUILD_CMD ${CMAKE_COMMAND} -E env MAKEFLAGS=$ENV{MAKEFLAGS} ${NCCL_BUILD_CMD})
  endif()
  ExternalProject_Add(${NCCL_NAME}
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/deps/${NCCL_NAME}
    PREFIX ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}
    INSTALL_DIR ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}
    BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/lib/libnccl${LIBEXT}
    INSTALL_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${NCCL_BUILD_CMD}
    BUILD_IN_SOURCE 1
  )

  ExternalProject_Get_Property(${NCCL_NAME} INSTALL_DIR)
  message(STATUS "NCCL install dir: ${INSTALL_DIR}")
  list(APPEND FLEXFLOW_INCLUDE_DIRS
    ${INSTALL_DIR}/include)
  list(APPEND FLEXFLOW_EXT_LIBRARIES
    ${INSTALL_DIR}/lib/libnccl${LIBEXT})
  set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/lib/")
  
  install(DIRECTORY ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/include/ DESTINATION include)
  install(DIRECTORY ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/lib/ DESTINATION lib PATTERN "pkgconfig" EXCLUDE)
endif()
