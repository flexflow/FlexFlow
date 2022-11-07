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
	# Check availability of precompiled Legion library
	set(LEGION_URL "")
	if(FF_USE_PRECOMPILED_LIBRARIES)
		if(LINUX_VERSION MATCHES "20.04")
		  if (CUDA_VERSION VERSION_EQUAL "11.0")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-20.04_11.0.3.tar.gz")
		  elseif(CUDA_VERSION VERSION_EQUAL "11.1")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-20.04_11.1.1.tar.gz")
		  elseif(CUDA_VERSION VERSION_EQUAL "11.2")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-20.04_11.2.2.tar.gz")
		  elseif(CUDA_VERSION VERSION_EQUAL "11.3")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-20.04_11.3.1.tar.gz")
		  elseif(CUDA_VERSION VERSION_EQUAL "11.4")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-20.04_11.4.3.tar.gz")
		  elseif(CUDA_VERSION VERSION_EQUAL "11.5")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-20.04_11.5.2.tar.gz")
		  elseif(CUDA_VERSION VERSION_EQUAL "11.6")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-20.04_11.6.2.tar.gz")
		  elseif(CUDA_VERSION VERSION_EQUAL "11.7")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-20.04_11.7.0.tar.gz")
		  endif()
		elseif(LINUX_VERSION MATCHES "18.04")
		  if (CUDA_VERSION VERSION_EQUAL "10.1")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-18.04_10.1.243.tar.gz")
		  elseif (CUDA_VERSION VERSION_EQUAL "10.2")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-18.04_10.2.89.tar.gz")
		  elseif (CUDA_VERSION VERSION_EQUAL "11.0")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-18.04_11.0.3.tar.gz")
		  elseif(CUDA_VERSION VERSION_EQUAL "11.1")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-18.04_11.1.1.tar.gz")
		  elseif(CUDA_VERSION VERSION_EQUAL "11.2")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-18.04_11.2.2.tar.gz")
		  elseif(CUDA_VERSION VERSION_EQUAL "11.3")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-18.04_11.3.1.tar.gz")
		  elseif(CUDA_VERSION VERSION_EQUAL "11.4")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-18.04_11.4.3.tar.gz")
		  elseif(CUDA_VERSION VERSION_EQUAL "11.5")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-18.04_11.5.2.tar.gz")
		  elseif(CUDA_VERSION VERSION_EQUAL "11.6")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-18.04_11.6.2.tar.gz")
		  elseif(CUDA_VERSION VERSION_EQUAL "11.7")
		    set(LEGION_URL "https://github.com/flexflow/flexflow-third-party/releases/latest/download/legion_ubuntu-18.04_11.7.0.tar.gz")
		  endif()
		endif()
	endif()

	if(LEGION_URL AND NOT FF_USE_GASNET)
		# Download and import pre-compiled Legion library
		message(STATUS "Using pre-compiled Legion library")
		message(STATUS "LEGION_URL: ${LEGION_URL}")
		set(LEGION_LIBRARY legion)
		set(REALM_LIBRARY realm)
		set(LEGION_DOWNLOAD legion)

		set(LEGION_TARBALL_PATH ${CMAKE_BINARY_DIR}/deps/${LEGION_DOWNLOAD}.tar.gz)
		set(LEGION_EXTRACTED_TARBALL_PATH ${CMAKE_BINARY_DIR}/build/export/${LEGION_DOWNLOAD})
		set(LEGION_FOLDER_PATH ${CMAKE_BINARY_DIR}/deps/${LEGION_DOWNLOAD})
		# If LEGION_FOLDER_PATH already exists (this is the case when calling `make install`), don't re-download.
		if(NOT EXISTS ${LEGION_FOLDER_PATH}/download_succeeded)
			file(DOWNLOAD ${LEGION_URL} ${LEGION_TARBALL_PATH} STATUS LEGION_DOWNLOAD_RESULT)
			list(GET LEGION_DOWNLOAD_RESULT 0 LEGION_DOWNLOAD_FAILED)

			if(LEGION_DOWNLOAD_FAILED)
				message(WARNING "Could not download ${LEGION_URL}! Building Legion library from scratch")
				set(LEGION_URL "")
			else()
				execute_process(
					COMMAND ${CMAKE_COMMAND} -E tar xzf ${LEGION_TARBALL_PATH}
					WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
				)
				execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${LEGION_EXTRACTED_TARBALL_PATH} ${LEGION_FOLDER_PATH})
				execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/build)
				execute_process(COMMAND ${CMAKE_COMMAND} -E remove ${LEGION_TARBALL_PATH})

				if(NOT EXISTS ${LEGION_FOLDER_PATH})
					message(WARNING "Could not extract tarball ${LEGION_TARBALL_PATH} to ${LEGION_FOLDER_PATH}! Building Legion library from scratch")
					set(LEGION_URL "")
				endif()
				execute_process(COMMAND ${CMAKE_COMMAND} -E touch ${LEGION_FOLDER_PATH}/download_succeeded)
			endif()
		endif()

		# If the download and extraction of the tarball succeeded, use the precompiled Legion library.
		if(LEGION_URL)
			SET(LEGION_INCLUDE_DIR ${LEGION_FOLDER_PATH}/include)
			SET(LEGION_DEFINE_DIR ${LEGION_INCLUDE_DIR})
			SET(LEGION_BIN_DIR ${LEGION_FOLDER_PATH}/bin/)
			SET(LEGION_LIB_DIR ${LEGION_FOLDER_PATH}/lib)
			SET(LEGION_SHARE_DIR ${LEGION_FOLDER_PATH}/share/)
			message(STATUS "Legion library path: ${LEGION_FOLDER_PATH}")

			add_library(${LEGION_LIBRARY} SHARED IMPORTED)
			add_library(${REALM_LIBRARY} SHARED IMPORTED)
			set_target_properties(${LEGION_LIBRARY} PROPERTIES IMPORTED_LOCATION ${LEGION_LIB_DIR}/liblegion${LIBEXT})
			set_target_properties(${REALM_LIBRARY} PROPERTIES IMPORTED_LOCATION ${LEGION_LIB_DIR}/librealm${LIBEXT})
		
			list(APPEND FLEXFLOW_INCLUDE_DIRS ${LEGION_INCLUDE_DIR} ${LEGION_INCLUDE_DIR}/hip_cuda_compat ${LEGION_INCLUDE_DIR}/legion ${LEGION_INCLUDE_DIR}/mappers ${LEGION_INCLUDE_DIR}/mathtypes ${LEGION_INCLUDE_DIR}/realm)
			
			install(DIRECTORY ${LEGION_SHARE_DIR} DESTINATION share)
			install(DIRECTORY ${LEGION_BIN_DIR} DESTINATION bin)
			install(DIRECTORY ${LEGION_LIB_DIR}/ DESTINATION lib)
		endif()
	endif()
	if(NOT LEGION_URL)
		# Build Legion from source
		message(STATUS "Building Legion from source")
		if(FF_USE_PYTHON)
		  set(Legion_USE_Python ON CACHE BOOL "enable Legion_USE_Python")
		endif()
		if(FF_USE_GASNET)
		  set(Legion_EMBED_GASNet ON CACHE BOOL "Use embed GASNet")
		  set(Legion_EMBED_GASNet_VERSION "GASNet-2022.3.0" CACHE STRING "GASNet version")
		  set(Legion_NETWORKS "gasnetex" CACHE STRING "GASNet conduit")
		  set(GASNet_CONDUIT ${FF_GASNET_CONDUIT})
		endif()
		message(STATUS "GASNET ROOT: $ENV{GASNet_ROOT_DIR}")
		set(Legion_MAX_DIM ${FF_MAX_DIM} CACHE STRING "Maximum number of dimensions")
		set(Legion_USE_CUDA ON CACHE BOOL "enable Legion_USE_CUDA")
		set(Legion_CUDA_ARCH ${FF_CUDA_ARCH} CACHE STRING "Legion CUDA ARCH")
		add_subdirectory(deps/legion)
		set(LEGION_LIBRARY Legion)
		set(LEGION_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/deps/legion/runtime)
		set(LEGION_DEFINE_DIR ${CMAKE_BINARY_DIR}/deps/legion/runtime)
	endif()
endif()
