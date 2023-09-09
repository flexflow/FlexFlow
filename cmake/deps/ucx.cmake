if("${FF_UCX_URL}" STREQUAL "")
    set(UCX_URL "https://github.com/openucx/ucx/releases/download/v1.14.0-rc1/ucx-1.14.0.tar.gz")
else()
    set(UCX_URL "${FF_UCX_URL}")
endif()

set(UCX_DIR ${CMAKE_CURRENT_BINARY_DIR}/ucx)
get_filename_component(UCX_COMPRESSED_FILE_NAME "${UCX_URL}" NAME)
# message(STATUS "UCX_URL: ${UCX_URL}")
# message(STATUS "UCX_COMPRESSED_FILE_NAME: ${UCX_COMPRESSED_FILE_NAME}")
set(UCX_COMPRESSED_FILE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${UCX_COMPRESSED_FILE_NAME}")
set(UCX_BUILD_NEEDED OFF)
set(UCX_CONFIG_FILE ${UCX_DIR}/config.txt)
set(UCX_BUILD_OUTPUT ${UCX_DIR}/build.log)

if(EXISTS ${UCX_CONFIG_FILE})
    file(READ ${UCX_CONFIG_FILE} PREV_UCX_CONFIG)
    # message(STATUS "PREV_UCX_CONFIG: ${PREV_UCX_CONFIG}")
    if("${UCX_URL}" STREQUAL "${PREV_UCX_CONFIG}")
	# configs match - no build needed
	set(UCX_BUILD_NEEDED OFF)
    else()
	    message(STATUS "UCX configuration has changed - rebuilding...")
	set(UCX_BUILD_NEEDED ON)
    endif()
else()
    message(STATUS "Configuring and building UCX...")
    set(UCX_BUILD_NEEDED ON)
endif()

if(UCX_BUILD_NEEDED)
    if(NOT EXISTS "${UCX_COMPRESSED_FILE_PATH}")
	message(STATUS "Downloading openucx/ucx from: ${UCX_URL}")
	file(
	    DOWNLOAD
	    "${UCX_URL}" "${UCX_COMPRESSED_FILE_PATH}"
	    SHOW_PROGRESS
	    STATUS status
	    LOG log
	)

	list(GET status 0 status_code)
	list(GET status 1 status_string)

	if(status_code EQUAL 0)
	    message(STATUS "Downloading... done")
	else()
	    message(FATAL_ERROR "error: downloading '${UCX_URL}' failed
		status_code: ${status_code}
		status_string: ${status_string}
		log:
		--- LOG BEGIN ---
		${log}
		--- LOG END ---"
	    )
	endif()
    else()
	message(STATUS "${UCX_COMPRESSED_FILE_NAME} already exists")
    endif()

    execute_process(COMMAND mkdir -p ${UCX_DIR})
    execute_process(COMMAND tar xzf ${UCX_COMPRESSED_FILE_PATH} -C ${UCX_DIR} --strip-components 1)
    message(STATUS "Building UCX...")
    execute_process(
	COMMAND sh -c "cd ${UCX_DIR} && ${UCX_DIR}/contrib/configure-release --prefix=${UCX_DIR}/install --enable-mt && make -j8 && make install"
	RESULT_VARIABLE UCX_BUILD_STATUS
	OUTPUT_FILE ${UCX_BUILD_OUTPUT}
	ERROR_FILE ${UCX_BUILD_OUTPUT}
    )

    if(UCX_BUILD_STATUS)
	message(FATAL_ERROR "UCX build result = ${UCX_BUILD_STATUS} - see ${UCX_BUILD_OUTPUT} for more details")
    endif()

    # Currently, we use default build configurations for UCX and therefore only save URL as configuration settings
    file(WRITE ${UCX_CONFIG_FILE} "${UCX_URL}")
endif()

if (FF_LEGION_NETWORKS STREQUAL "gasnet" AND FF_GASNET_CONDUIT STREQUAL "ucx")
    set(ENV{UCX_HOME} "${UCX_DIR}/install")
endif()

if (FF_LEGION_NETWORKS STREQUAL "ucx")
    set(ucx_DIR ${UCX_DIR}/cmake)
    set(ENV{Legion_NETWORKS} "ucx")
    message(STATUS "Legion_NETWORKS: $ENV{Legion_NETWORKS}")
endif()
