if(FF_USE_EXTERNAL_TENSORRT)
    if(NOT "${TENSORRT_ROOT}" STREQUAL "")
        set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${TENSORRT_ROOT}/lib/cmake)
    endif()
    find_package(TensorRT REQUIRED)
    get_target_property(TENSORRT_INCLUDE_DIRS TensorRT::nvinfer INTERFACE_INCLUDE_DIRECTORIES)
    string(REGEX REPLACE "/include" "" TENSORRT_ROOT_TMP ${TENSORRT_INCLUDE_DIRS})
    if("${TENSORRT_ROOT}" STREQUAL "")
        set(TENSORRT_ROOT ${TENSORRT_ROOT_TMP})
    else()
        if(NOT "${TENSORRT_ROOT}" STREQUAL ${TENSORRT_ROOT_TMP})
            message(FATAL_ERROR "TENSORRT_ROOT is not set correctly ${TENSORRT_ROOT} ${TENSORRT_ROOT_TMP}")
        endif()
    endif()
    message(STATUS "Use external TensorRT cmake found: ${TENSORRT_ROOT_TMP}")
    message(STATUS "Use external TensorRT: ${TENSORRT_ROOT}")
    set(TENSORRT_LIBRARY TensorRT::nvinfer)
else()
    # Check availability of precompiled TensorRT library
    set(TENSORRT_URL "")
    if((FF_USE_PREBUILT_TENSORRT OR FF_USE_ALL_PREBUILT_LIBRARIES) AND CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
        # For now, reusing pre-compiled TensorRT library only works for specific configurations.
        if(LINUX_VERSION MATCHES "20.04")
            set(TENSORRT_URL "https://developer.download.nvidia.com/compute/machine-learning/tensorrt/8.0.1/tars/TensorRT-8.0.1.6.Linux.x86_64-gnu.cuda-11.3.cudnn8.2.tar.gz")
        elseif(LINUX_VERSION MATCHES "18.04")
            set(TENSORRT_URL "https://developer.download.nvidia.com/compute/machine-learning/tensorrt/7.2.3.4/tars/TensorRT-7.2.3.4.Linux.x86_64-gnu.cuda-11.1.cudnn8.0.tar.gz")
        endif()
    endif()

    if(TENSORRT_URL)
        # Download and import pre-compiled TensorRT library
        message(STATUS "Using pre-compiled TensorRT library")
        message(STATUS "TENSORRT_URL: ${TENSORRT_URL}")
        set(TENSORRT_NAME tensorrt)
        set(TENSORRT_LIBRARY nvinfer)
        
        include(FetchContent)
        FetchContent_Declare(${TENSORRT_NAME}
            URL ${TENSORRT_URL}
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
        )
        FetchContent_GetProperties(${TENSORRT_NAME})
        if(NOT ${TENSORRT_NAME}_POPULATED)
            FetchContent_Populate(${TENSORRT_NAME})
        endif()
        
        set(TENSORRT_FOLDER_PATH ${${TENSORRT_NAME}_SOURCE_DIR}/TensorRT)
        SET(TENSORRT_INCLUDE_DIR ${TENSORRT_FOLDER_PATH}/include)
        SET(TENSORRT_LIB_DIR ${TENSORRT_FOLDER_PATH}/lib)
        message(STATUS "TensorRT library path: ${TENSORRT_FOLDER_PATH}")
        
        add_library(${TENSORRT_LIBRARY} SHARED IMPORTED)
        set_target_properties(${TENSORRT_LIBRARY} PROPERTIES IMPORTED_LOCATION ${TENSORRT_LIB_DIR}/libnvinfer.so)
    
        list(APPEND FLEXFLOW_INCLUDE_DIRS 
            ${TENSORRT_INCLUDE_DIR}
        )
        
        install(DIRECTORY ${TENSORRT_INCLUDE_DIR} DESTINATION include)
        install(DIRECTORY ${TENSORRT_LIB_DIR}/ DESTINATION lib)
    else()
        # Message to indicate that TensorRT needs to be built or handled separately.
        message(FATAL_ERROR "TensorRT is not available and needs to be built or downloaded manually.")
    endif()
endif()

# Add TensorRT to the FlexFlow build
if(TENSORRT_LIBRARY)
    list(APPEND FF_CC_FLAGS -DUSE_TENSORRT)
    list(APPEND FF_NVCC_FLAGS -DUSE_TENSORRT)
    target_include_directories(flexflow PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(flexflow PRIVATE ${TENSORRT_LIBRARY})
endif()
