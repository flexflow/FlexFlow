find_package(NCCL REQUIRED)
#set(NCCL_NAME nccl_internal)
## set(NCCL_CUDA_ARCH "-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
## message("NCCL_CUDA_ARCH: ${NCCL_CUDA_ARCH}")
#
#if (FF_USE_PREBUILT_NCCL OR FF_USE_ALL_PREBUILT_LIBRARIES)
#  if(NCCL_PATH)
#    set(NCCL_ROOT ${NCCL_PATH})
#  else()
#    # if NCCL_PATH is not set, let's try to find it in the CUDA root
#    set(NCCL_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
#  endif()
#  
#  find_library(NCCL_LIBRARY
#    NAMES libnccl${LIBEXT}
#    PATHS ${NCCL_ROOT} ${CUDA_ROOT}
#    PATH_SUFFIXES lib lib64
#    DOC "NCCL library." )
#
#  find_path(NCCL_INCLUDE_DIR
#    NAMES nccl.h
#    HINTS ${NCCL_ROOT}
#    PATH_SUFFIXES include 
#    DOC "NCCL include directory.")
#  
#  # find NCCL, set NCCL lib and include    
#  if(NCCL_LIBRARY AND NCCL_INCLUDE_DIR)
#    set(NCCL_FOUND ON)
#    set(NCCL_LIBRARIES ${NCCL_LIBRARY})
#    set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
#  endif()
#  
#  # find NCCL
#  if(NCCL_FOUND)
#    message( STATUS "NCCL include : ${NCCL_INCLUDE_DIRS}" )
#    message( STATUS "NCCL libraries : ${NCCL_LIBRARIES}" )
#    add_library(nccl SHARED IMPORTED)
#  
#  # Build NCCL from source
#else()
#    message(STATUS "Building NCCL from source")
#    list(TRANSFORM CUDA_GENCODE PREPEND "NVCC_GENCODE=" OUTPUT_VARIABLE NCCL_BUILD_NVCC_GENCODE)
#  
#    ExternalProject_Add(${NCCL_NAME}
#      SOURCE_DIR ${PROJECT_SOURCE_DIR}/deps/${NCCL_NAME}
#      PREFIX ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}
#      INSTALL_DIR ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}
#      BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/lib/libnccl${LIBEXT}
#      INSTALL_COMMAND ""
#      CONFIGURE_COMMAND ""
#      BUILD_COMMAND make src.build "${NCCL_BUILD_NVCC_GENCODE}" "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}" "BUILDDIR=${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}"
#      BUILD_IN_SOURCE 1
#    )
#
#    ExternalProject_Get_Property(${NCCL_NAME} INSTALL_DIR)
#    message(STATUS "NCCL install dir: ${INSTALL_DIR}")
#    set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/lib/")
#    
#    install(DIRECTORY ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/include/ DESTINATION include)
#    install(DIRECTORY ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/lib/ DESTINATION lib PATTERN "pkgconfig" EXCLUDE)
#  endif()
#
#  # Download and import pre-compiled NCCL library
#  message(STATUS "Using pre-compiled NCCL library")
#  message(STATUS "NCCL_URL: ${NCCL_URL}")
#
#  include(FetchContent)
#  FetchContent_Declare(${NCCL_NAME}
#    URL ${NCCL_URL}
#    CONFIGURE_COMMAND ""
#    BUILD_COMMAND ""
#  )
#  FetchContent_GetProperties(${NCCL_NAME})
#  if(NOT ${NCCL_NAME}_POPULATED)
#    FetchContent_Populate(${NCCL_NAME})
#  endif()
#  
#  set(NCCL_FOLDER_PATH ${${NCCL_NAME}_SOURCE_DIR}/deps/nccl)
#  set(NCCL_INCLUDE_DIR ${NCCL_FOLDER_PATH}/include)
#  set(NCCL_LIB_DIR ${NCCL_FOLDER_PATH}/lib)
#  message(STATUS "NCCL library path: ${NCCL_FOLDER_PATH}")
#  add_library(nccl SHARED IMPORTED)
#  set_target_properties(nccl PROPERTIES IMPORTED_LOCATION ${NCCL_FOLDER_PATH})
#
#  list(APPEND FLEXFLOW_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
#  list(APPEND FLEXFLOW_EXT_LIBRARIES ${NCCL_LIB_DIR}/libnccl${LIBEXT})
#  install(DIRECTORY ${NCCL_INCLUDE_DIR}/ DESTINATION include)
#  install(DIRECTORY ${NCCL_LIB_DIR}/ DESTINATION lib PATTERN "pkgconfig" EXCLUDE)
#    
#  set(NCCL_LIB "${INSTALL_DIR}/lib/libnccl${LIBEXT}")
#else()
#  # Build NCCL from source
#  message(STATUS "Building NCCL from source")
#  list(TRANSFORM CUDA_GENCODE PREPEND "NVCC_GENCODE=" OUTPUT_VARIABLE NCCL_BUILD_NVCC_GENCODE)
#
#  include(ExternalProject)
#  ExternalProject_Add(${NCCL_NAME}
#   SOURCE_DIR ${PROJECT_SOURCE_DIR}/deps/nccl
#   PREFIX ${CMAKE_BINARY_DIR}/deps/nccl
#   INSTALL_DIR ${CMAKE_BINARY_DIR}/deps/nccl
#   BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/deps/nccl/lib/libnccl${LIBEXT}
#   INSTALL_COMMAND ""
#   CONFIGURE_COMMAND ""
#   BUILD_COMMAND make src.build "${NCCL_BUILD_NVCC_GENCODE}" "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}" "BUILDDIR=${CMAKE_BINARY_DIR}/deps/nccl" "CXX=${CMAKE_CXX_COMPILER} -w" CC="${CMAKE_CC_COMPILER}"
#   BUILD_IN_SOURCE 1
#  )
#
#  ExternalProject_Get_Property(${NCCL_NAME} INSTALL_DIR)
#  message(STATUS "NCCL install dir: ${INSTALL_DIR}")
#  set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${CMAKE_BINARY_DIR}/deps/nccl/lib/")
#
#  set(NCCL_INCLUDE_DIR "${INSTALL_DIR}/include")
#  set(NCCL_LIB "${INSTALL_DIR}/lib/libnccl${LIBEXT}")
#endif()
#message("NCCL_LIB = ${NCCL_LIB}")
#message("INSTALL_DIR = ${INSTALL_DIR}")
#
#add_library(nccl INTERFACE)
#target_include_directories(nccl SYSTEM INTERFACE ${NCCL_INCLUDE_DIRS})
#add_dependencies(nccl ${NCCL_NAME})
#target_link_libraries(nccl INTERFACE ${NCCL_LIB})
