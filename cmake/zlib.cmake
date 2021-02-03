find_package(ZLIB REQUIRED)
if(ZLIB_FOUND)
  list(APPEND FLEXFLOW_EXT_LIBRARIES 
    ${ZLIB_LIBRARIES})
  message( STATUS "ZLIB libraries : ${ZLIB_LIBRARIES}" )
else()
  message( FATAL_ERROR "ZLIB package not found")
endif()