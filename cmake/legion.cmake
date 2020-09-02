include(utils)
detect_installed_gpus(COMPUTE_CAPABILITY)
#message(${COMPUTE_CAPABILITY})

if (NOT ${FlexFlow_USE_External_Legion})
  set(LEGION_SRC_DIR ${PROJECT_SOURCE_DIR}/legion)
else()
  message( STATUS "Use External Legion : ${EXTERNAL_LEGION_DIR}" )
  set(LEGION_SRC_DIR ${EXTERNAL_LEGION_DIR})
endif()

set(LEGION_NAME project_legion)

# TODO fix python version
message(STATUS "Building ${LEGION_NAME}")
ExternalProject_Add(${LEGION_NAME}
 SOURCE_DIR ${LEGION_SRC_DIR}
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
   -DLegion_USE_Python=${FlexFlow_USE_Python}
   -DLegion_Python_Version=3.6
   <SOURCE_DIR>
)

ExternalProject_get_property(${LEGION_NAME} INSTALL_DIR)
include_directories(${INSTALL_DIR}/include ${INSTALL_DIR}/include/mappers)
add_library(legion STATIC IMPORTED)
set_property(TARGET legion PROPERTY IMPORTED_LOCATION ${INSTALL_DIR}/lib64/liblegion.a)
add_library(realm STATIC IMPORTED)
set_property(TARGET realm PROPERTY IMPORTED_LOCATION ${INSTALL_DIR}/lib64/librealm.a)

#install(DIRECTORY ${INSTALL_DIR}/ DESTINATION ${INSTALL_DIR} USE_SOURCE_PERMISSIONS)