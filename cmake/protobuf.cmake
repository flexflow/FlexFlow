list(APPEND CMAKE_PREFIX_PATH ${PROTOBUF_ROOT})
find_package(Protobuf REQUIRED)

if(Protobuf_FOUND)
  list(APPEND FLEXFLOW_EXT_LIBRARIES ${Protobuf_LIBRARIES})
  include_directories(${Protobuf_INCLUDE_DIRS})
  message( STATUS "Protobuf version : ${Protobuf_VERSION}" )
  message( STATUS "Protobuf include path : ${Protobuf_INCLUDE_DIRS}" )
  message( STATUS "Protobuf libraries : ${Protobuf_LIBRARIES}" )
  message( STATUS "Protobuf compiler : ${Protobuf_PROTOC_EXECUTABLE}" )
else()
  message( WARNING "Protobuf package not found -> specify search path via PROTOBUF_ROOT variable")
endif()


