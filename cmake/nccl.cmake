set(NCCL_NAME nccl)
# set(NCCL_CUDA_ARCH "-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
# message("NCCL_CUDA_ARCH: ${NCCL_CUDA_ARCH}")

set(NCCL_URL "")
if(FF_USE_PRECOMPILED_LIBRARIES)
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

  set(NCCL_TARBALL_PATH ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}.tar.gz)
  set(NCCL_EXTRACTED_TARBALL_PATH ${CMAKE_BINARY_DIR}/build/deps/${NCCL_NAME})
  set(NCCL_FOLDER_PATH ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME})
  file(DOWNLOAD ${NCCL_URL} ${NCCL_TARBALL_PATH} STATUS NCCL_DOWNLOAD_RESULT)
  list(GET NCCL_DOWNLOAD_RESULT 0 NCCL_DOWNLOAD_FAILED)

  if(NCCL_DOWNLOAD_FAILED)
    #message(STATUS "Could not download prebuilt library. (${NCCL_DOWNLOAD_RESULT})")
    #file(REMOVE ${NCCL_TARBALL_PATH})
    message(FATAL_ERROR "Could not download ${NCCL_URL}!")
  endif()

  if(EXISTS ${NCCL_FOLDER_PATH})
    message(FATAL_ERROR "${NCCL_FOLDER_PATH} already exists!")
  endif()

  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${NCCL_TARBALL_PATH}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${NCCL_EXTRACTED_TARBALL_PATH} ${NCCL_FOLDER_PATH})
  execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/build)
  execute_process(COMMAND ${CMAKE_COMMAND} -E remove ${NCCL_TARBALL_PATH})

  if(NOT EXISTS ${NCCL_FOLDER_PATH})
    message(FATAL_ERROR "Could not extract tarball ${NCCL_TARBALL_PATH} to ${NCCL_FOLDER_PATH}!")
  endif()

  add_library(nccl SHARED IMPORTED)
  set_target_properties(nccl PROPERTIES IMPORTED_LOCATION ${NCCL_FOLDER_PATH})
  set(NCCL_INCLUDE_DIR ${NCCL_FOLDER_PATH}/include)
  set(NCCL_LIB_DIR ${NCCL_FOLDER_PATH}/lib)

  list(APPEND FLEXFLOW_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
  list(APPEND FLEXFLOW_EXT_LIBRARIES ${NCCL_LIB_DIR}/libnccl${LIBEXT})
  install(DIRECTORY ${NCCL_INCLUDE_DIR} DESTINATION include)
	install(DIRECTORY ${NCCL_LIB_DIR}/ DESTINATION lib)
else()
  # Build NCCL from source
  list(TRANSFORM CUDA_GENCODE PREPEND "NVCC_GENCODE=" OUTPUT_VARIABLE NCCL_BUILD_NVCC_GENCODE)
  
  ExternalProject_Add(${NCCL_NAME}
   SOURCE_DIR ${PROJECT_SOURCE_DIR}/deps/${NCCL_NAME}
   PREFIX ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}
   INSTALL_DIR ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}
   BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/lib/libnccl${LIBEXT}
   INSTALL_COMMAND ""
   CONFIGURE_COMMAND ""
   BUILD_COMMAND make src.build "${NCCL_BUILD_NVCC_GENCODE}" "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}" "BUILDDIR=${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}"
   BUILD_IN_SOURCE 1
  )

  ExternalProject_Get_Property(${NCCL_NAME} INSTALL_DIR)
  message(STATUS "NCCL install dir: ${INSTALL_DIR}")
  list(APPEND FLEXFLOW_INCLUDE_DIRS
    ${INSTALL_DIR}/include)
  list(APPEND FLEXFLOW_EXT_LIBRARIES
    ${INSTALL_DIR}/lib/libnccl${LIBEXT})
  set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/lib/")

endif()
