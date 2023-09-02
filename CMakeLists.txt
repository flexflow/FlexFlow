cmake_minimum_required(VERSION 3.10)
project(FlexFlow)


include(ExternalProject)

# Set policy CMP0074 to eliminate cmake warnings
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0077 NEW)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  # Fix DOWNLOAD_EXTRACT_TIMESTAMP warnings
  cmake_policy(SET CMP0135 NEW)
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)
set(FLEXFLOW_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS} -fPIC -UNDEBUG")

option(INFERENCE_TESTS "Run inference tests" OFF)
set(LIBTORCH_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch" CACHE STRING "LibTorch Path")
if (INFERENCE_TESTS)
  find_package(Torch REQUIRED PATHS ${LIBTORCH_PATH} NO_DEFAULT_PATH)
  set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS} -fPIC ${TORCH_CXX_FLAGS}")
  message(STATUS "LIBTORCH_PATH: ${LIBTORCH_PATH}")
  message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
endif()

# Set a default build type if none was specified
set(default_build_type "Debug")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
endif()

# do not disable assertions even if in release mode
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -UNDEBUG")

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(LIBEXT ".so")
endif()

# only used for pypi
option(FF_BUILD_FROM_PYPI "Build from pypi" OFF)

# build shared or static flexflow lib
option(BUILD_SHARED_LIBS "Build shared libraries instead of static ones" ON)

# option for using Python
option(FF_USE_PYTHON "Enable Python" ON)

# option to download pre-compiled NCCL/Legion libraries
option(FF_USE_PREBUILT_NCCL "Enable use of NCCL pre-compiled library, if available" ON)
option(FF_USE_PREBUILT_LEGION "Enable use of Legion pre-compiled library, if available" ON)
option(FF_USE_ALL_PREBUILT_LIBRARIES "Enable use of all pre-compiled libraries, if available" OFF)

# option for using Python
set(FF_GASNET_CONDUITS aries udp mpi ibv ucx)
set(FF_GASNET_CONDUIT "mpi" CACHE STRING "Select GASNet conduit ${FF_GASNET_CONDUITS}")
set_property(CACHE FF_GASNET_CONDUIT PROPERTY STRINGS ${FF_GASNET_CONDUITS})
set(FF_LEGION_NETWORKS "" CACHE STRING "Network backend(s) to use")

if ((FF_LEGION_NETWORKS STREQUAL "gasnet" AND FF_GASNET_CONDUIT STREQUAL "ucx") OR FF_LEGION_NETWORKS STREQUAL "ucx")
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
        install(DIRECTORY ${UCX_DIR}/install/bin/ DESTINATION bin)
        install(DIRECTORY ${UCX_DIR}/install/include/ DESTINATION include)
        install(DIRECTORY ${UCX_DIR}/install/lib/ DESTINATION lib)
        install(DIRECTORY ${UCX_DIR}/install/share/ DESTINATION share)
    endif()

    if (FF_LEGION_NETWORKS STREQUAL "ucx")
        set(ucx_DIR ${UCX_DIR}/cmake)
        set(ENV{Legion_NETWORKS} "ucx")
        message(STATUS "Legion_NETWORKS: $ENV{Legion_NETWORKS}")
    endif()
else()
    message(STATUS "FF_GASNET_CONDUIT: ${FF_GASNET_CONDUIT}")
endif()

set(FF_GPU_BACKENDS cuda hip_cuda hip_rocm intel)
set(FF_GPU_BACKEND "cuda" CACHE STRING "Select GPU Backend ${FF_GPU_BACKENDS}")
set_property(CACHE FF_GPU_BACKEND PROPERTY STRINGS ${FF_GPU_BACKENDS})

# option for cuda arch
set(FF_CUDA_ARCH "autodetect" CACHE STRING "Target CUDA Arch")
if ((FF_GPU_BACKEND STREQUAL "cuda" OR FF_GPU_BACKEND STREQUAL "hip_cuda") AND FF_CUDA_ARCH STREQUAL "")
  message(FATAL_ERROR "FF_CUDA_ARCH cannot be an empty string. Set it to `autodetect`, `all`, or pass one or multiple valid CUDA archs.")
endif()
# option for hip arch
set(FF_HIP_ARCH "all" CACHE STRING "Target HIP Arch")
if (FF_GPU_BACKEND STREQUAL "hip_rocm" AND FF_CUDA_ARCH STREQUAL "")
  message(FATAL_ERROR "FF_HIP_ARCH cannot be an empty string. Set it to `all`, or pass one or multiple valid HIP archs.")
endif()

# option for nccl
option(FF_USE_NCCL "Run FlexFlow with NCCL" OFF)

# option for avx2
option(FF_USE_AVX2 "Run FlexFlow with AVX2" OFF)

# option for max dim
set(FF_MAX_DIM "4" CACHE STRING "Maximum dimention of tensors")

# option for legion
option(FF_USE_EXTERNAL_LEGION "Use pre-installed Legion" OFF)
set(LEGION_MAX_RETURN_SIZE "32768" CACHE STRING "Maximum Legion return size")

set(FLEXFLOW_EXT_LIBRARIES "")
set(FLEXFLOW_INCLUDE_DIRS "")

# get FLAGS from ENV
set(CC_FLAGS $ENV{CC_FLAGS})
set(NVCC_FLAGS $ENV{NVCC_FLAGS})
set(LD_FLAGS $ENV{LD_FLAGS})

# Set global FLAGS
list(APPEND CC_FLAGS
  -std=c++17)
list(APPEND NVCC_FLAGS
  -std=c++17)

add_compile_options(${CC_FLAGS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${NVCC_FLAGS})
link_libraries(${LD_FLAGS})

# Detect OS type and Linux version (if it applies)
set(LINUX_VERSION "")
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  find_program(LSB_RELEASE_EXEC lsb_release)
  if(LSB_RELEASE_EXEC)
    execute_process(COMMAND ${LSB_RELEASE_EXEC} -r --short 
                    OUTPUT_VARIABLE LINUX_VERSION 
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Linux Version: ${LINUX_VERSION}")
  endif()
endif()

# Detect CPU architecture
message(STATUS "CPU architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}")

if(FF_GPU_BACKEND STREQUAL "hip_cuda" OR FF_GPU_BACKEND STREQUAL "hip_rocm")
  set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
endif()

# CUDA
if (FF_GPU_BACKEND STREQUAL "cuda" OR FF_GPU_BACKEND STREQUAL "hip_cuda")
  include(cuda)
endif()

# HIP
if (FF_GPU_BACKEND STREQUAL "hip_rocm" OR FF_GPU_BACKEND STREQUAL "hip_cuda")
  include(hip)
endif()

# CUDNN
if (FF_GPU_BACKEND STREQUAL "cuda" OR FF_GPU_BACKEND STREQUAL "hip_cuda")
  include(cudnn)
endif()

# NCCL
if(FF_USE_NCCL)
  if(FF_GPU_BACKEND STREQUAL "hip_cuda" OR FF_GPU_BACKEND STREQUAL "cuda")
    include(nccl)
  endif() 
  list(APPEND FF_CC_FLAGS
    -DFF_USE_NCCL)
  list(APPEND FF_NVCC_FLAGS
    -DFF_USE_NCCL)
endif()

# Inference tests
if(INFERENCE_TESTS)
  list(APPEND FF_CC_FLAGS
    -DINFERENCE_TESTS)
  list(APPEND FF_NVCC_FLAGS
    -DINFERENCE_TESTS)
endif()

# Legion
include(legion)

# json
include(json)

# variant
include(variant)

# optional
include(optional)

if (FF_GPU_BACKEND STREQUAL "cuda")
  list(APPEND FF_CC_FLAGS
    -DFF_USE_CUDA)
  list(APPEND FF_NVCC_FLAGS
    -DFF_USE_CUDA)
elseif (FF_GPU_BACKEND STREQUAL "hip_cuda")
  list(APPEND FF_CC_FLAGS
    -DFF_USE_HIP_CUDA)
  list(APPEND FF_HIPCC_FLAGS
    -DFF_USE_HIP_CUDA)
elseif (FF_GPU_BACKEND STREQUAL "hip_rocm")
  list(APPEND FF_CC_FLAGS
    -DFF_USE_HIP_ROCM)
  list(APPEND FF_HIPCC_FLAGS
    -DFF_USE_HIP_ROCM)
else()
endif()

# Start build FlexFlow
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND FF_CC_FLAGS
    -DFF_DEBUG)
  list(APPEND FF_NVCC_FLAGS
    -DFF_DEBUG)
endif()

message(STATUS "FlexFlow MAX_DIM: ${FF_MAX_DIM}")
message(STATUS "LEGION_MAX_RETURN_SIZE: ${LEGION_MAX_RETURN_SIZE}")

list(APPEND FF_CC_FLAGS
  -DMAX_TENSOR_DIM=${FF_MAX_DIM}
  -DLEGION_MAX_RETURN_SIZE=${LEGION_MAX_RETURN_SIZE})

if(FF_USE_AVX2)
  list(APPEND FF_CC_FLAGS
    -DFF_USE_AVX2
    -mavx2)
endif()

list(APPEND FF_NVCC_FLAGS
  -Wno-deprecated-gpu-targets
  -DMAX_TENSOR_DIM=${FF_MAX_DIM}
  -DLEGION_MAX_RETURN_SIZE=${LEGION_MAX_RETURN_SIZE})

list(APPEND FF_LD_FLAGS
  -lrt
  -ldl
  -rdynamic
  -lstdc++fs)

# Set FF FLAGS
add_compile_options(${FF_CC_FLAGS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${FF_NVCC_FLAGS} -UNDEBUG)
link_libraries(${FF_LD_FLAGS})

list(APPEND FLEXFLOW_INCLUDE_DIRS
  ${FLEXFLOW_ROOT}/include
  ${FLEXFLOW_ROOT})

file(GLOB_RECURSE FLEXFLOW_HDR
  LIST_DIRECTORIES False
  ${FLEXFLOW_ROOT}/include/*.h)
  list(APPEND FLEXFLOW_HDR ${FLEXFLOW_ROOT}/inference/file_loader.h)

file(GLOB_RECURSE FLEXFLOW_SRC
  LIST_DIRECTORIES False
  ${FLEXFLOW_ROOT}/src/*.cc)
list(REMOVE_ITEM FLEXFLOW_SRC "${FLEXFLOW_ROOT}/src/runtime/cpp_driver.cc")
list(APPEND FLEXFLOW_SRC ${FLEXFLOW_ROOT}/inference/file_loader.cc)

set(FLEXFLOW_CPP_DRV_SRC
  ${FLEXFLOW_ROOT}/src/runtime/cpp_driver.cc)

add_library(substitution_loader SHARED
  ${FLEXFLOW_ROOT}/src/runtime/substitution_loader.cc)
target_include_directories(substitution_loader PRIVATE ${FLEXFLOW_INCLUDE_DIRS})
target_link_libraries(substitution_loader nlohmann_json::nlohmann_json)


#message("FLEXFLOW_INCLUDE_DIRS: ${FLEXFLOW_INCLUDE_DIRS}")

# compile flexflow lib
if (FF_GPU_BACKEND STREQUAL "cuda")
  file(GLOB_RECURSE FLEXFLOW_GPU_SRC
    LIST_DIRECTORIES False
    ${FLEXFLOW_ROOT}/src/*.cu)

  add_compile_definitions(FF_USE_CUDA)

  if(BUILD_SHARED_LIBS)
    cuda_add_library(flexflow SHARED ${FLEXFLOW_GPU_SRC} ${FLEXFLOW_SRC} OPTIONS ${CUDA_GENCODE})
  else()
    cuda_add_library(flexflow STATIC ${FLEXFLOW_GPU_SRC} ${FLEXFLOW_SRC} OPTIONS ${CUDA_GENCODE})
  endif()
elseif(FF_GPU_BACKEND STREQUAL "hip_cuda" OR FF_GPU_BACKEND STREQUAL "hip_rocm")
  file(GLOB_RECURSE FLEXFLOW_GPU_SRC
    LIST_DIRECTORIES False
    ${FLEXFLOW_ROOT}/src/*.cpp)

  if(BUILD_SHARED_LIBS)
    add_library(flexflow SHARED ${FLEXFLOW_GPU_SRC} ${FLEXFLOW_SRC})
  else()
    add_library(flexflow STATIC ${FLEXFLOW_GPU_SRC} ${FLEXFLOW_SRC})
  endif()

  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})

  find_package(hip REQUIRED)

  if (FF_GPU_BACKEND STREQUAL "hip_cuda")
    # The targets defined by the hip cmake config only target amd devices.
    # For targeting nvidia devices, we'll make our own interface target,
    # hip_device_nvidia, that includes the rocm and hip headers. 
    add_library(hip_device_nvidia INTERFACE)

    if (NOT FF_CUDA_ARCH STREQUAL "")
      target_compile_options(hip_device_nvidia INTERFACE -arch=compute_${FF_CUDA_ARCH})
    endif()

    target_include_directories(hip_device_nvidia SYSTEM INTERFACE ${HIP_INCLUDE_DIRS} ${ROCM_PATH}/include)
    target_include_directories(hip_device_nvidia INTERFACE ${HIP_INCLUDE_DIRS} ${ROCM_PATH}/include)

    add_compile_definitions(FF_USE_HIP_CUDA)

    # Linking cuda: 
    # We do not explicitly link cuda. hipcc when targeting nvidia will 
    # use nvcc under the hood. nvcc when used for linking will handle 
    # linking cuda dependencies
    target_link_libraries(flexflow hip_device_nvidia)
  elseif(FF_GPU_BACKEND STREQUAL "hip_rocm")
    find_package(hipblas REQUIRED)
    find_package(miopen REQUIRED)
    if(FF_USE_NCCL)
      find_package(rccl REQUIRED)
    endif()
    # find_package(rocrand REQUIRED)
    find_library(HIP_RAND_LIBRARY hiprand REQUIRED)

    add_compile_definitions(FF_USE_HIP_ROCM)

    if (FF_HIP_ARCH STREQUAL "")
      message(FATAL_ERROR "FF_HIP_ARCH is undefined")
    endif()
    set_property(TARGET flexflow PROPERTY HIP_ARCHITECTURES "${HIP_ARCH_LIST}")

    message(STATUS "FF_GPU_BACKEND: ${FF_GPU_BACKEND}")
    message(STATUS "FF_HIP_ARCH: ${FF_HIP_ARCH}")
    message(STATUS "HIP_ARCH_LIST: ${HIP_ARCH_LIST}")
    get_property(CHECK_HIP_ARCHS TARGET flexflow PROPERTY HIP_ARCHITECTURES)
    message(STATUS "CHECK_HIP_ARCHS: ${CHECK_HIP_ARCHS}")
    message(STATUS "HIP_CLANG_PATH: ${HIP_CLANG_PATH}")

    # The hip cmake config module defines three targets, 
    # hip::amdhip64, hip::host, and hip::device.
    #
    # hip::host and hip::device are interface targets. hip::amdhip64 is an 
    # imported target for libamdhip.
    #
    # You do not directly link to hip::amdhip64. hip::host links to hip::amdhip64
    # and hip::device links to hip::host. Link to hip::host to just use hip without 
    # compiling any GPU code. Link to hip::device to compile the GPU device code.
    #
    # Docs (outdated):
    # https://rocmdocs.amd.com/en/latest/Installation_Guide/Using-CMake-with-AMD-ROCm.html
    target_link_libraries(flexflow hip::device roc::hipblas MIOpen ${HIP_RAND_LIBRARY})
    if(FF_USE_NCCL)
        target_link_libraries(flexflow rccl)
    endif()
  endif()
else()
  message(FATAL_ERROR "Unsupported FF_GPU_BACKEND for cmake: ${FF_GPU_BACKEND}")
endif()

if(FF_USE_NCCL AND (FF_GPU_BACKEND STREQUAL "hip_cuda" OR FF_GPU_BACKEND STREQUAL "cuda"))
  add_dependencies(flexflow ${NCCL_NAME})
endif()

target_include_directories(flexflow PUBLIC ${FLEXFLOW_INCLUDE_DIRS})
# LEGION_URL is defined if we found a precompiled Legion library to download
if(LEGION_URL)
  # Legion builds produce two library files: one for the Legion runtime and one for the Realm runtime. 
  # When linking FlexFlow to a precompiled version of Legion, we need to manually link to both library files.
  target_link_libraries(flexflow ${LEGION_LIBRARY} ${REALM_LIBRARY} ${FLEXFLOW_EXT_LIBRARIES} nlohmann_json::nlohmann_json mpark_variant optional)
  add_dependencies(flexflow ${LEGION_NAME})
else()
  # When building Legion from source, we do so by calling add_subdirectory(), and obtain a library with both the
  # Legion and Realm runtimes. The library's name is saved into the LEGION_LIBRARY variable. Hence, we only need
  # to link FlexFlow to ${LEGION_LIBRARY}
  target_link_libraries(flexflow ${LEGION_LIBRARY} ${FLEXFLOW_EXT_LIBRARIES} nlohmann_json::nlohmann_json mpark_variant optional)
endif()

#library api version, bump from time to time
set(SOVERSION 1)

set_target_properties(flexflow PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(flexflow PROPERTIES OUTPUT_NAME "flexflow${INSTALL_SUFFIX}")
set_target_properties(flexflow PROPERTIES SOVERSION ${SOVERSION})
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(flexflow PROPERTIES BUILD_RPATH "\$ORIGIN")
  set_target_properties(flexflow PROPERTIES INSTALL_RPATH "\$ORIGIN")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set_target_properties(flexflow PROPERTIES BUILD_RPATH "@loader_path")
  set_target_properties(flexflow PROPERTIES INSTALL_RPATH "@loader_path")
endif()

# python related
if (FF_USE_PYTHON)
  # create flexflow_cffi_header.py
  add_custom_command(TARGET flexflow
    PRE_BUILD	
    COMMAND ${FLEXFLOW_ROOT}/python/flexflow_cffi_build.py --ffhome-dir ${FLEXFLOW_ROOT} --output-dir ${FLEXFLOW_ROOT}/python/flexflow/core
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Creating flexflow_cffi_header.py..."
  )
  if (NOT FF_BUILD_FROM_PYPI)
    # generate the Legion Python bindings library. When building from pip, we need to do this post-install to prevent Legion from overwriting the path to the Legion shared library
    add_custom_command(TARGET flexflow
      POST_BUILD	
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/deps/legion/bindings/python/setup.py build --cmake-build-dir ${Legion_BINARY_DIR}/runtime --prefix ${Legion_BINARY_DIR} --build-lib=${Legion_BINARY_DIR}/bindings/python ${Legion_PYTHON_EXTRA_INSTALL_ARGS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/deps/legion/bindings/python
    )
    # create flexflow_python interpreter. When building from pip, we install the FF_HOME/python/flexflow_python script instead.
    add_custom_command(TARGET flexflow
      PRE_BUILD	
      COMMAND ${PYTHON_EXECUTABLE} ${FLEXFLOW_ROOT}/python/flexflow_python_build.py --build-dir ${CMAKE_BINARY_DIR}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Creating flexflow_python interpreter..."
    )
    install(PROGRAMS ${CMAKE_BINARY_DIR}/flexflow_python DESTINATION "bin")
  endif()
endif()

if (INFERENCE_TESTS)
  target_link_libraries(flexflow "${TORCH_LIBRARIES}")
  set_property(TARGET flexflow PROPERTY CXX_STANDARD 14)
endif()

# build binary
option(FF_BUILD_TOKENIZER "build tokenizer=cpp for LLM serving" ON)
option(FF_BUILD_RESNET "build resnet example" OFF)
option(FF_BUILD_RESNEXT "build resnext example" OFF)
option(FF_BUILD_ALEXNET "build alexnet example" OFF)
option(FF_BUILD_DLRM "build DLRM example" OFF)
option(FF_BUILD_XDL "build XDL example" OFF)
option(FF_BUILD_INCEPTION "build inception example" OFF)
option(FF_BUILD_CANDLE_UNO "build candle uno example" OFF)
option(FF_BUILD_TRANSFORMER "build transformer example" OFF)
option(FF_BUILD_MOE "build mixture of experts example" OFF)
option(FF_BUILD_MLP_UNIFY "build mlp unify example" OFF)
option(FF_BUILD_SPLIT_TEST "build split test example" OFF)
option(FF_BUILD_SPLIT_TEST_2 "build split test 2 example" OFF)
option(FF_BUILD_MLP_UNIFY_INFERENCE "build mlp unify inference example" OFF)
option(FF_BUILD_ALL_INFERENCE_EXAMPLES "build all inference examples. Overrides others" OFF)
option(FF_BUILD_ALL_EXAMPLES "build all examples. Overrides others" OFF)
option(FF_BUILD_UNIT_TESTS "build non-operator unit tests" OFF)
option(FF_BUILD_SUBSTITUTION_TOOL "build substitution conversion tool" OFF)
option(FF_BUILD_VISUALIZATION_TOOL "build substitution visualization tool" OFF)

if(FF_BUILD_UNIT_TESTS)
  set(BUILD_GMOCK OFF)
  add_subdirectory(deps/googletest)
  enable_testing()
  add_subdirectory(tests/unit)
endif()

if(FF_BUILD_SUBSTITUTION_TOOL)
  add_subdirectory(tools/protobuf_to_json)
endif()

if(FF_BUILD_VISUALIZATION_TOOL)
  add_subdirectory(tools/substitutions_to_dot)
endif()

if(FF_BUILD_ALL_INFERENCE_EXAMPLES OR FF_BUILD_TOKENIZER)
  if (FF_GPU_BACKEND STREQUAL "hip_rocm")
    SET(SPM_USE_BUILTIN_PROTOBUF OFF CACHE BOOL "Use builtin version of protobuf to compile SentencePiece")
  endif()
  # Ensure Rust is installed
  execute_process(COMMAND rustc --version
                RESULT_VARIABLE RUST_COMMAND_RESULT
                OUTPUT_VARIABLE RUSTC_OUTPUT
                ERROR_QUIET)
  if(NOT RUST_COMMAND_RESULT EQUAL 0)
    message(FATAL_ERROR "Rust is not installed on the system. Please install it by running: 'curl https://sh.rustup.rs -sSf | sh -s -- -y' and following the instructions on the screen.")
  endif()
  # Ensure Cargo is installed
  execute_process(COMMAND cargo --version
                  RESULT_VARIABLE CARGO_RESULT
                  OUTPUT_QUIET ERROR_QUIET)
  if(NOT CARGO_RESULT EQUAL 0)
    message(FATAL_ERROR "Rust is installed, but cargo is not. Please install it by running: 'curl https://sh.rustup.rs -sSf | sh -s -- -y' and following the instructions on the screen.")
  endif()
  add_subdirectory(deps/tokenizers-cpp tokenizers EXCLUDE_FROM_ALL)
  target_include_directories(flexflow PUBLIC deps/tokenizers-cpp/include)
  target_link_libraries(flexflow tokenizers_cpp)
endif()
if(FF_BUILD_RESNET OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/ResNet)
endif()

if(FF_BUILD_RESNEXT OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/resnext50)
endif()

if(FF_BUILD_ALEXNET OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/AlexNet)
endif()

if(FF_BUILD_MLP_UNIFY OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/MLP_Unify)
endif()

if(FF_BUILD_SPLIT_TEST OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/split_test)
endif()

if(FF_BUILD_SPLIT_TEST_2 OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/split_test_2)
endif()

if(FF_BUILD_INCEPTION OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/InceptionV3)
endif()

#TODO: Once functional add to BUILD_ALL_EXAMPLES
if(FF_BUILD_CANDLE_UNO OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/candle_uno)
endif()

if(FF_BUILD_DLRM OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/DLRM)

  #add_executable(generate_dlrm_hetero_strategy src/runtime/dlrm_strategy_hetero.cc)
  #target_include_directories(generate_dlrm_hetero_strategy PUBLIC ${FLEXFLOW_INCLUDE_DIRS})

  #add_executable(generate_dlrm_strategy src/runtime/dlrm_strategy.cc)
  #target_include_directories(generate_dlrm_strategy PUBLIC ${FLEXFLOW_INCLUDE_DIRS})
endif()

if(FF_BUILD_XDL OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/XDL)
endif()

if(FF_BUILD_TRANSFORMER OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/Transformer)
endif()

if(FF_BUILD_MOE OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(examples/cpp/mixture_of_experts)
endif()

if(FF_BUILD_ALL_INFERENCE_EXAMPLES OR FF_BUILD_ALL_EXAMPLES)
  add_subdirectory(inference/spec_infer)
  add_subdirectory(inference/incr_decoding)
endif()


# installation
set(INCLUDE_DEST "include")
set(LIB_DEST "lib")
install(FILES ${FLEXFLOW_HDR} DESTINATION ${INCLUDE_DEST})
install(TARGETS flexflow DESTINATION ${LIB_DEST})
# install python
if (FF_USE_PYTHON)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils import sysconfig; print(sysconfig.get_python_lib(plat_specific=False,standard_lib=False))" OUTPUT_VARIABLE PY_DEST OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (NOT FF_BUILD_FROM_PYPI)
    install(
      DIRECTORY ${FLEXFLOW_ROOT}/python/flexflow/
      DESTINATION ${PY_DEST}/flexflow
      FILES_MATCHING 
      PATTERN "*.py")
  else()
    # pip automatically installs all *.py files in the python/flexflow folder, but because flexflow_cffi_header.py is generated at build time, we have to install it manually.
    install(
      PROGRAMS ${FLEXFLOW_ROOT}/python/flexflow/core/flexflow_cffi_header.py
      DESTINATION ${PY_DEST}/flexflow/core
    )
    # Use setup.py script to re-install the Python bindings library with the right library paths. 
    # Need to put the instructions in a subfolder because of issue below:
    # https://stackoverflow.com/questions/43875499/do-post-processing-after-make-install-in-cmake
    add_subdirectory(cmake/pip_install)
  endif()
endif()
