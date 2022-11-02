if(FF_USE_EXTERNAL_LEGION)
	if(NOT "${LEGION_ROOT}" STREQUAL "")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${LEGION_ROOT}/share/Legion/cmake)
	endif()
	find_package(Legion REQUIRED)
	get_target_property(LEGION_INCLUDE_DIRS Legion::RealmRuntime INTERFACE_INCLUDE_DIRECTORIES)
	string(REGEX REPLACE "/include" "" LEGION_ROOT_TMP ${LEGION_INCLUDE_DIRS})
	if("${LEGION_ROOT}" STREQUAL "")
		set(LEGION_ROOT ${LEGION_ROOT_TMP})
	else()
		if(NOT "${LEGION_ROOT}" STREQUAL ${LEGION_ROOT_TMP})
			message( FATAL_ERROR "LEGION_ROOT is not set correctly ${LEGION_ROOT} ${LEGION_ROOT_TMP}")
		endif()
	endif()
	message(STATUS "Use external Legion cmake found: ${LEGION_ROOT_TMP}")
	message(STATUS "Use external Legion: ${LEGION_ROOT}")
	set(LEGION_LIBRARY Legion::Legion)
else()
	# if(FF_USE_PYTHON)
	#   set(Legion_USE_Python ON CACHE BOOL "enable Legion_USE_Python")
	# endif()
	# if(FF_USE_GASNET)
	#   set(Legion_EMBED_GASNet ON CACHE BOOL "Use embed GASNet")
	#   set(Legion_EMBED_GASNet_VERSION "GASNet-2022.3.0" CACHE STRING "GASNet version")
	#   set(Legion_NETWORKS "gasnetex" CACHE STRING "GASNet conduit")
	#   set(GASNet_CONDUIT ${FF_GASNET_CONDUIT})
	# endif()
	# message(STATUS "GASNET ROOT: $ENV{GASNet_ROOT_DIR}")
	# set(Legion_MAX_DIM ${FF_MAX_DIM} CACHE STRING "Maximum number of dimensions")
	# set(Legion_USE_CUDA ON CACHE BOOL "enable Legion_USE_CUDA")
	# set(Legion_CUDA_ARCH ${FF_CUDA_ARCH} CACHE STRING "Legion CUDA ARCH")
	# add_subdirectory(deps/legion)
	# set(LEGION_LIBRARY Legion)

	set(LEGION_LIBRARY legion)
	set(REALM_LIBRARY realm)
	set(LEGION_DOWNLOAD legionDownload)
	set(LEGION_URL https://www.gabrieleoliaro.it/legion.tar.gz)

	ExternalProject_Add(${LEGION_DOWNLOAD}
	  SOURCE_DIR ""
	  PREFIX ${CMAKE_BINARY_DIR}/deps/${LEGION_LIBRARY}
	  URL ${LEGION_URL}
	  DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/deps/
	  #INSTALL_DIR ${CMAKE_BINARY_DIR}/deps/${LEGION_LIBRARY}
	  #BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/deps/${LEGION_LIBRARY}/lib/liblegion${LIBEXT} ${CMAKE_BINARY_DIR}/deps/${LEGION_LIBRARY}/lib/librealm${LIBEXT}
	  INSTALL_COMMAND ""
	  CONFIGURE_COMMAND ""
	  BUILD_COMMAND ""
	  UPDATE_COMMAND ""
	)

	ExternalProject_Get_Property(${LEGION_DOWNLOAD} INSTALL_DIR)
	message(STATUS "LEGION install dir: ${INSTALL_DIR}")

	SET(LEGION_BASE_DIR ${INSTALL_DIR}/src/${LEGION_LIBRARY})
	SET(LEGION_INCLUDE_DIR ${LEGION_BASE_DIR}/include)
	SET(LEGION_BIN_DIR ${LEGION_BASE_DIR}/bin/)
	SET(LEGION_LIB_DIR ${LEGION_BASE_DIR}/lib)
	SET(LEGION_SHARE_DIR ${LEGION_BASE_DIR}/share/)

	add_library(${LEGION_LIBRARY} STATIC IMPORTED)
	add_library(${REALM_LIBRARY} STATIC IMPORTED)
	set_target_properties(${LEGION_LIBRARY} PROPERTIES IMPORTED_LOCATION ${LEGION_LIB_DIR}/liblegion${LIBEXT})
	set_target_properties(${REALM_LIBRARY} PROPERTIES IMPORTED_LOCATION ${LEGION_LIB_DIR}/librealm${LIBEXT})
	set_target_properties(${LEGION_LIBRARY} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${LEGION_INCLUDE_DIR})

	#add_library(${LEGION_LIBRARY} SHARED IMPORTED)
	#set_target_properties(${LEGION_LIBRARY} PROPERTIES IMPORTED_LOCATION ${LEGION_LIB_DIR}/liblegion${LIBEXT} ${LEGION_LIB_DIR}/librealm${LIBEXT})
	list(APPEND FLEXFLOW_INCLUDE_DIRS ${LEGION_INCLUDE_DIR})
	#list(APPEND FLEXFLOW_EXT_LIBRARIES ${LEGION_LIB_DIR}/liblegion${LIBEXT} ${LEGION_LIB_DIR}/librealm${LIBEXT})

	#include_directories(${LEGION_INCLUDE_DIR})


	install(DIRECTORY ${LEGION_SHARE_DIR} DESTINATION share)
	install(DIRECTORY ${LEGION_BIN_DIR} DESTINATION bin)


endif()

