# set(NCCL_CUDA_ARCH "-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
# message("NCCL_CUDA_ARCH: ${NCCL_CUDA_ARCH}")
#list(TRANSFORM CUDA_GENCODE PREPEND "NVCC_GENCODE=" OUTPUT_VARIABLE NCCL_BUILD_NVCC_GENCODE)

set(NCCL_NAME nccl)
set(NCCL_URL https://github.com/gabrieleoliaro/flexflow-third-party/releases/latest/download/nccl_11.1.1_ubuntu-20.04.tar.gz)

ExternalProject_Add(${NCCL_NAME}
  SOURCE_DIR ""
  PREFIX ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}
  URL ${NCCL_URL}
  DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/deps/
  INSTALL_DIR ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}
  BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/lib/libnccl${LIBEXT}
  INSTALL_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  BUILD_IN_SOURCE 1
)

ExternalProject_Get_Property(${NCCL_NAME} INSTALL_DIR)
message(STATUS "NCCL install dir: ${INSTALL_DIR}")
list(APPEND FLEXFLOW_INCLUDE_DIRS
  ${INSTALL_DIR}/include)
list(APPEND FLEXFLOW_EXT_LIBRARIES
  ${INSTALL_DIR}/lib/libnccl${LIBEXT})
set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/lib/")

# ExternalProject_Add(${NCCL_NAME}
#  SOURCE_DIR ${PROJECT_SOURCE_DIR}/deps/${NCCL_NAME}
#  PREFIX ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}
#  INSTALL_DIR ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}
#  BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/lib/libnccl${LIBEXT}
#  INSTALL_COMMAND ""
#  CONFIGURE_COMMAND ""
#  BUILD_COMMAND make src.build "${NCCL_BUILD_NVCC_GENCODE}" "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}" "BUILDDIR=${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}"
#  BUILD_IN_SOURCE 1
# )

# ExternalProject_Get_Property(${NCCL_NAME} INSTALL_DIR)
# message(STATUS "NCCL install dir: ${INSTALL_DIR}")
# list(APPEND FLEXFLOW_INCLUDE_DIRS
#   ${INSTALL_DIR}/include)
# list(APPEND FLEXFLOW_EXT_LIBRARIES
#   ${INSTALL_DIR}/lib/libnccl${LIBEXT})
# set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/lib/")

# include(FetchContent)
# FetchContent_Declare(nccl URL )
# FetchContent_MakeAvailable(nccl)

# list(APPEND FLEXFLOW_INCLUDE_DIRS
#   ${nccl_SOURCE_DIR}/include)
# list(APPEND FLEXFLOW_EXT_LIBRARIES
#   ${nccl_SOURCE_DIR}/lib/libnccl${LIBEXT})

