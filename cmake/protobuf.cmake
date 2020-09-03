if(NOT ${FlexFlow_USE_External_Protobuf})
  #include(FindProtobuf)
  set(PROTOBUF_NAME ext_protobuf)
  set(PROTOBUF_AUTOGEN_NAME ext_protobuf_autogen)
  
  # hack: PROTOBUF_AUTOGEN_NAME is a sudo project only for autogen as it has to be run in source dir (BUILD_IN_SOURCE 1)
  message(STATUS "Building ${PROTOBUF_NAME} ${PROJECT_SOURCE_DIR}/protobuf2")
  ExternalProject_Add(${PROTOBUF_AUTOGEN_NAME}
   SOURCE_DIR ${PROJECT_SOURCE_DIR}/protobuf2
   PREFIX ${PROTOBUF_AUTOGEN_NAME}
   CONFIGURE_COMMAND <SOURCE_DIR>/autogen.sh
   BUILD_COMMAND echo "sudo make"
   INSTALL_COMMAND echo "sudo make install"
   BUILD_IN_SOURCE 1
  )
  ExternalProject_Add(${PROTOBUF_NAME}
   SOURCE_DIR ${PROJECT_SOURCE_DIR}/protobuf2
   PREFIX ${PROTOBUF_NAME}
   INSTALL_DIR ${PROTOBUF_NAME}/install
   CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> "CC=${CMAKE_C_COMPILER}" "CXX=${CMAKE_CXX_COMPILER}"
  )
  add_dependencies(${PROTOBUF_NAME} ${PROTOBUF_AUTOGEN_NAME})
    
  ExternalProject_get_property(${PROTOBUF_NAME} INSTALL_DIR)
  set(Protobuf_FOUND ON)
  set(Protobuf_INCLUDE_DIRS ${INSTALL_DIR}/include)
  set(Protobuf_VERSION 0.0)
  set(Protobuf_LIBRARIES ${INSTALL_DIR}/lib/libprotobuf${LIBEXT} -lpthread)
  set(Protobuf_PROTOC_EXECUTABLE ${INSTALL_DIR}/bin/protoc) 
  
  install(DIRECTORY ${INSTALL_DIR}/ DESTINATION ${CMAKE_INSTALL_PREFIX} USE_SOURCE_PERMISSIONS)
  
  function(protobuf_generate_cpp SRCS HDRS)
    if(NOT ARGN)
      message(SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called without any proto files")
      return()
    endif()
    set(${SRCS})
    set(${HDRS})
    foreach(FIL ${ARGN})
      get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
      get_filename_component(FIL_WE ${FIL} NAME_WE)
      get_filename_component(DIR_FIL ${FIL} DIRECTORY)
      list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc")
      list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h")
      add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc"
               "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h"
        COMMAND  ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} -I ${DIR_FIL} ${ABS_FIL}
        DEPENDS ${ABS_FIL} ${Protobuf_PROTOC_EXECUTABLE}
        COMMENT "Running C++ protocol buffer compiler on ${FIL}"
        VERBATIM )
    endforeach()
    set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
    set(${SRCS} ${${SRCS}} PARENT_SCOPE)
    set(${HDRS} ${${HDRS}} PARENT_SCOPE)
  endfunction()

  # set(protobuf_BUILD_TESTS OFF CACHE BOOL "Disable tests for protobuf")
  # set(BUILD_SHARED_LIBS OFF)
  # set(LIBRARY_POLICY STATIC)
  # add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../protobuf/cmake ${CMAKE_CURRENT_BINARY_DIR}/protobuf)
  # include_directories(${CMAKE_CURRENT_LIST_DIR}/../protobuf/src)
  # link_directories(${CMAKE_CURRENT_BINARY_DIR}/protobuf)
  # set(protobuf_lib_name protobuf)
  # if(CMAKE_BUILD_TYPE MATCHES "Debug")
  #   set(protobuf_lib_name protobufd)
  # endif()
else()
  set(PROTOBUF_ROOT ${EXTERNAL_PROTOBUF_DIR})
  list(APPEND CMAKE_PREFIX_PATH ${PROTOBUF_ROOT})
  set(protobuf_lib_name protobuf)
  find_package(Protobuf REQUIRED)
endif()

if(Protobuf_FOUND)
  list(APPEND FLEXFLOW_EXT_LIBRARIES 
    ${Protobuf_LIBRARIES})
  message( STATUS "Protobuf version : ${Protobuf_VERSION}" )
  message( STATUS "Protobuf include path : ${Protobuf_INCLUDE_DIRS}" )
  message( STATUS "Protobuf libraries : ${Protobuf_LIBRARIES}" )
  message( STATUS "Protobuf compiler : ${Protobuf_PROTOC_EXECUTABLE}" )
else()
  message( WARNING "Protobuf package not found -> specify search path via PROTOBUF_ROOT variable")
endif()
include_directories(${Protobuf_INCLUDE_DIRS})


