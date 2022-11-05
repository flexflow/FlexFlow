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
	set(LEGION_URL https://github.com/gabrieleoliaro/flexflow-third-party/releases/latest/download/legion_11.1.1_ubuntu-20.04.tar.gz)

	ExternalProject_Add(${LEGION_DOWNLOAD}
		PREFIX ${CMAKE_BINARY_DIR}/deps/${LEGION_LIBRARY}
		SOURCE_DIR ""
		URL ${LEGION_URL}
		DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/deps/
		BUILD_COMMAND ""
		#BUILD_COMMAND "echo 'Building...'"
		BUILD_BYPRODUCTS deps/${LEGION_LIBRARY}/src/${LEGION_DOWNLOAD}/lib/liblegion${LIBEXT} deps/${LEGION_LIBRARY}/src/${LEGION_DOWNLOAD}/lib/librealm${LIBEXT}
		INSTALL_COMMAND ""
		#INSTALL_DIR ${CMAKE_BINARY_DIR}/deps/${LEGION_LIBRARY}
		CONFIGURE_COMMAND ""
		UPDATE_COMMAND ""
	)

	ExternalProject_Get_Property(${LEGION_DOWNLOAD} INSTALL_DIR)
	message(STATUS "LEGION install dir: ${INSTALL_DIR}")

	SET(LEGION_BASE_DIR ${INSTALL_DIR}/src/${LEGION_DOWNLOAD})
	SET(LEGION_INCLUDE_DIR ${LEGION_BASE_DIR}/include)
	SET(LEGION_BIN_DIR ${LEGION_BASE_DIR}/bin/)
	SET(LEGION_LIB_DIR ${LEGION_BASE_DIR}/lib)
	SET(LEGION_SHARE_DIR ${LEGION_BASE_DIR}/share/)

	message(STATUS "LEGION_BASE_DIR: ${LEGION_BASE_DIR}")
	message(STATUS "LEGION_INCLUDE_DIR: ${LEGION_INCLUDE_DIR}")
	message(STATUS "LEGION_BIN_DIR: ${LEGION_BIN_DIR}")
	message(STATUS "LEGION_LIB_DIR: ${LEGION_LIB_DIR}")
	message(STATUS "LEGION_SHARE_DIR: ${LEGION_SHARE_DIR}")

	add_library(${LEGION_LIBRARY} SHARED IMPORTED)
	add_library(${REALM_LIBRARY} SHARED IMPORTED)
	set_target_properties(${LEGION_LIBRARY} PROPERTIES IMPORTED_LOCATION ${LEGION_LIB_DIR}/liblegion${LIBEXT})
	set_target_properties(${REALM_LIBRARY} PROPERTIES IMPORTED_LOCATION ${LEGION_LIB_DIR}/librealm${LIBEXT})
	#set_target_properties(${LEGION_LIBRARY} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${LEGION_INCLUDE_DIR})

	#add_library(${LEGION_LIBRARY} SHARED IMPORTED)
	list(APPEND FLEXFLOW_INCLUDE_DIRS ${LEGION_INCLUDE_DIR} ${LEGION_INCLUDE_DIR}/hip_cuda_compat ${LEGION_INCLUDE_DIR}/legion ${LEGION_INCLUDE_DIR}/mappers ${LEGION_INCLUDE_DIR}/mathtypes ${LEGION_INCLUDE_DIR}/realm)
	#list(APPEND FLEXFLOW_EXT_LIBRARIES ${LEGION_LIB_DIR}/liblegion${LIBEXT} ${LEGION_LIB_DIR}/librealm${LIBEXT})

	#include_directories(${LEGION_INCLUDE_DIR})


	install(DIRECTORY ${LEGION_SHARE_DIR} DESTINATION share)
	install(DIRECTORY ${LEGION_BIN_DIR} DESTINATION bin)
	install(DIRECTORY ${LEGION_LIB_DIR}/ DESTINATION lib)


endif()

