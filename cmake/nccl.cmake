include(aliasing)

add_library(nccl INTERFACE)

if (FF_USE_EXTERNAL_NCCL)
  find_package(NCCL REQUIRED)
else()
  message(STATUS "Building NCCL from source")
  list(TRANSFORM CUDA_GENCODE PREPEND "NVCC_GENCODE=" OUTPUT_VARIABLE NCCL_BUILD_NVCC_GENCODE)

  ExternalProject_Add(nccl_source_build
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/deps/${NCCL_NAME}
    PREFIX ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}
    INSTALL_DIR ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}
    BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}/lib/libnccl${LIBEXT}
    INSTALL_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make src.build "${NCCL_BUILD_NVCC_GENCODE}" "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}" "BUILDDIR=${CMAKE_BINARY_DIR}/deps/${NCCL_NAME}"
    BUILD_IN_SOURCE 1
  )

  ExternalProject_Get_Property(nccl_source_build INSTALL_DIR)
  set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES "${CMAKE_BINARY_DIR}/deps/nccl_source_build/lib/")
  
  install(DIRECTORY ${CMAKE_BINARY_DIR}/deps/nccl_source_build/include/ DESTINATION include)
  install(DIRECTORY ${CMAKE_BINARY_DIR}/deps/nccl_source_build/lib/ DESTINATION lib PATTERN "pkgconfig" EXCLUDE)

  set(NCCL_INCLUDE_DIR "${INSTALL_DIR}/include")
  set(NCCL_LIBRARIES "${INSTALL_DIR}/lib/libnccl${LIBEXT}")

  add_dependencies(nccl nccl_source_build)
endif()

message(STATUS "NCCL_LIBRARIES = ${NCCL_LIBRARIES}")
target_include_directories(nccl SYSTEM INTERFACE ${NCCL_INCLUDE_DIRS})
target_link_libraries(nccl INTERFACE ${NCCL_LIBRARIES})
