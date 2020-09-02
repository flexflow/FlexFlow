# include(utils)
# detect_installed_gpus(COMPUTE_CAPABILITY)
# message(${COMPUTE_CAPABILITY})

# option(Legion_USE_CUDA "ENABLE CUDA WITH LEGION" ON)
# option(Legion_BUILD_APPS "BUILD LEGION APPS" ON)
# set(Legion_MAX_DIM 4 CACHE STRING "Maximum number of dimensions")
# set(LG_RT_DIR CACHE STRING "${CMAKE_CURRENT_BINARY_DIR}/legion/runtime/")
# set(Legion_CUDA_ARCH ${COMPUTE_CAPABILITY})
# message( WARNING "legion dir: ${CMAKE_CURRENT_LIST_DIR}")
# add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../legion ${CMAKE_CURRENT_BINARY_DIR}/legion)
# if (NOT ${FLOW_USE_EXTERNAL_LEGION})
# endif()

set(LEGION_NAME project_legion)

message(STATUS "Building ${LEGION_NAME}")
ExternalProject_Add(${LEGION_NAME}
 SOURCE_DIR ${PROJECT_SOURCE_DIR}/legion-cr
 PREFIX ${LEGION_NAME}
 INSTALL_DIR ${LEGION_NAME}/install
 CONFIGURE_COMMAND ${CMAKE_COMMAND}
   -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
   -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
   -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF
   -DLegion_USE_CUDA=ON
   -DLegion_MAX_DIM=4
   -DLegion_BUILD_EXAMPLES=OFF
   -DLegion_BUILD_APPS=OFF
   -DLegion_CUDA_ARCH=60
   -DLegion_USE_Python=ON
   -DLegion_Python_Version=3.6
   <SOURCE_DIR>
)

ExternalProject_get_property(${LEGION_NAME} INSTALL_DIR)
include_directories(${INSTALL_DIR}/include ${INSTALL_DIR}/include/mappers)
add_library(legion STATIC IMPORTED)
set_property(TARGET legion PROPERTY IMPORTED_LOCATION ${INSTALL_DIR}/lib64/liblegion.a)
add_library(realm STATIC IMPORTED)
set_property(TARGET realm PROPERTY IMPORTED_LOCATION ${INSTALL_DIR}/lib64/librealm.a)

install(DIRECTORY ${INSTALL_DIR}/ DESTINATION ${INSTALL_DIR} USE_SOURCE_PERMISSIONS)