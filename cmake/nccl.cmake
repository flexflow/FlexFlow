set(NCCL_NAME nccl_internal)
# set(NCCL_CUDA_ARCH "-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
# message("NCCL_CUDA_ARCH: ${NCCL_CUDA_ARCH}")

set(NCCL_URL "")
if((FF_USE_PREBUILT_NCCL OR FF_USE_ALL_PREBUILT_LIBRARIES) AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
  if(LINUX_VERSION MATCHES "20.04")
    if (CUDA_VERSION VERSION_EQUAL "11.0")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-20.04_11.0.3.tar.gz")
    elseif(CUDA_VERSION VERSION_EQUAL "11.1")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-20.04_11.1.1.tar.gz")
    elseif(CUDA_VERSION VERSION_EQUAL "11.2")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-20.04_11.2.2.tar.gz")
    elseif(CUDA_VERSION VERSION_EQUAL "11.3")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-20.04_11.3.1.tar.gz")
    elseif(CUDA_VERSION VERSION_EQUAL "11.4")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-20.04_11.4.3.tar.gz")
    elseif(CUDA_VERSION VERSION_EQUAL "11.5")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-20.04_11.5.2.tar.gz")
    elseif(CUDA_VERSION VERSION_EQUAL "11.6")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-20.04_11.6.2.tar.gz")
    elseif(CUDA_VERSION VERSION_EQUAL "11.7")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-20.04_11.7.0.tar.gz")
    endif()
  elseif(LINUX_VERSION MATCHES "18.04")
    if (CUDA_VERSION VERSION_EQUAL "10.1")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-18.04_10.1.243.tar.gz")
    elseif (CUDA_VERSION VERSION_EQUAL "10.2")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-18.04_10.2.89.tar.gz")
    elseif (CUDA_VERSION VERSION_EQUAL "11.0")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-18.04_11.0.3.tar.gz")
    elseif(CUDA_VERSION VERSION_EQUAL "11.1")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-18.04_11.1.1.tar.gz")
    elseif(CUDA_VERSION VERSION_EQUAL "11.2")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-18.04_11.2.2.tar.gz")
    elseif(CUDA_VERSION VERSION_EQUAL "11.3")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-18.04_11.3.1.tar.gz")
    elseif(CUDA_VERSION VERSION_EQUAL "11.4")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-18.04_11.4.3.tar.gz")
    elseif(CUDA_VERSION VERSION_EQUAL "11.5")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-18.04_11.5.2.tar.gz")
    elseif(CUDA_VERSION VERSION_EQUAL "11.6")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-18.04_11.6.2.tar.gz")
    elseif(CUDA_VERSION VERSION_EQUAL "11.7")
      set(NCCL_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/nccl_ubuntu-18.04_11.7.0.tar.gz")
    endif()
  endif()
endif()

if(NCCL_URL)
  # Download and import pre-compiled NCCL library
  message(STATUS "Using pre-compiled NCCL library")
  message(STATUS "NCCL_URL: ${NCCL_URL}")

  include(FetchContent)
  FetchContent_Declare(${NCCL_NAME}
    URL ${NCCL_URL}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
  )
  FetchContent_GetProperties(${NCCL_NAME})
  if(NOT ${NCCL_NAME}_POPULATED)
    FetchContent_Populate(${NCCL_NAME})
  endif()
  
  set(NCCL_FOLDER_PATH ${${NCCL_NAME}_SOURCE_DIR}/deps/nccl)
  set(NCCL_INCLUDE_DIR ${NCCL_FOLDER_PATH}/include)
  set(NCCL_LIB_DIR ${NCCL_FOLDER_PATH}/lib)
  message(STATUS "NCCL library path: ${NCCL_FOLDER_PATH}")
  add_library(nccl SHARED IMPORTED)
  set_target_properties(nccl PROPERTIES IMPORTED_LOCATION ${NCCL_FOLDER_PATH})

  list(APPEND FLEXFLOW_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
  list(APPEND FLEXFLOW_EXT_LIBRARIES ${NCCL_LIB_DIR}/libnccl${LIBEXT})
  install(DIRECTORY ${NCCL_INCLUDE_DIR}/ DESTINATION include)
  install(DIRECTORY ${NCCL_LIB_DIR}/ DESTINATION lib PATTERN "pkgconfig" EXCLUDE)
    
  set(NCCL_LIB "${INSTALL_DIR}/lib/libnccl${LIBEXT}")
else()
  # Build NCCL from source
  message(STATUS "Building NCCL from source")
  list(TRANSFORM CUDA_GENCODE PREPEND "NVCC_GENCODE=" OUTPUT_VARIABLE NCCL_BUILD_NVCC_GENCODE)

  include(ExternalProject)
  ExternalProject_Add(${NCCL_NAME}
   SOURCE_DIR ${PROJECT_SOURCE_DIR}/deps/nccl
   PREFIX ${CMAKE_BINARY_DIR}/deps/nccl
   INSTALL_DIR ${CMAKE_BINARY_DIR}/deps/nccl
   BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/deps/nccl/lib/libnccl${LIBEXT}
   INSTALL_COMMAND ""
   CONFIGURE_COMMAND ""
   BUILD_COMMAND make src.build "${NCCL_BUILD_NVCC_GENCODE}" "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}" "BUILDDIR=${CMAKE_BINARY_DIR}/deps/nccl" "CXX=${CMAKE_CXX_COMPILER} -w" CC="${CMAKE_CC_COMPILER}"
   BUILD_IN_SOURCE 1
  )

  ExternalProject_Get_Property(${NCCL_NAME} INSTALL_DIR)
  message(STATUS "NCCL install dir: ${INSTALL_DIR}")
  set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${CMAKE_BINARY_DIR}/deps/nccl/lib/")

  set(NCCL_INCLUDE_DIR "${INSTALL_DIR}/include")
  set(NCCL_LIB "${INSTALL_DIR}/lib/libnccl${LIBEXT}")
endif()
message("NCCL_LIB = ${NCCL_LIB}")
message("INSTALL_DIR = ${INSTALL_DIR}")

add_library(nccl INTERFACE)
target_include_directories(nccl SYSTEM INTERFACE ${NCCL_INCLUDE_DIR})
add_dependencies(nccl ${NCCL_NAME})
target_link_libraries(nccl INTERFACE ${NCCL_LIB})
