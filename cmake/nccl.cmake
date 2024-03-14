include(aliasing)
#set(NCCL_NAME nccl_internal)
## set(NCCL_CUDA_ARCH "-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
## message("NCCL_CUDA_ARCH: ${NCCL_CUDA_ARCH}")
#
if (FF_USE_EXTERNAL_NCCL)
  find_package(NCCL REQUIRED)

  alias_library(nccl NCCL)
else()
  message(STATUS "Building NCCL from source")
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
  set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/lib/")
  
  install(DIRECTORY ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/include/ DESTINATION include)
  install(DIRECTORY ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/lib/ DESTINATION lib PATTERN "pkgconfig" EXCLUDE)

  set(NCCL_INCLUDE_DIR "${INSTALL_DIR}/include")
  set(NCCL_LIB "${INSTALL_DIR}/lib/libnccl${LIBEXT}")

  message("NCCL_LIB = ${NCCL_LIB}")
  message("INSTALL_DIR = ${INSTALL_DIR}")

  add_library(nccl INTERFACE)
  target_include_directories(nccl SYSTEM INTERFACE ${NCCL_INCLUDE_DIRS})
  add_dependencies(nccl ${NCCL_NAME})
  target_link_libraries(nccl INTERFACE ${NCCL_LIB})
endif()
