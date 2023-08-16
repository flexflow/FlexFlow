# GASNet is a private dependency and only needs to be pulled in for static
# builds
set(Legion_NETWORKS )
set(Legion_EMBED_GASNet )
if("${Legion_NETWORKS}" MATCHES ".*gasnet(1|ex).*")
  if(NOT TARGET GASNet::GASNet)
    # define a GASNet::GASNet package that just points at:
    #  gasnet libs (embedded or external) for static builds
    #  external libs (e.g. mpi) needed in all cases
    cmake_minimum_required(VERSION 3.11)  # required for adding libs to imported target
    add_library(GASNet::GASNet INTERFACE IMPORTED)
    target_link_libraries(GASNet::GASNet INTERFACE )
  endif()
endif()

# LLVM is a private dependency and only needs to be pulled in for static
# builds
set(Legion_USE_LLVM OFF)
if((NOT OFF) AND Legion_USE_LLVM)
  set(Legion_LLVM_COMPONENTS )
  set(LLVM_CONFIG_EXECUTABLE )
  find_package(LLVM REQUIRED COMPONENTS ${Legion_LLVM_COMPONENTS})
endif()

# CUDA is only a private dependency of Legion but has a few usage requirements
# so we add it in here
set(Legion_USE_CUDA ON)
if(Legion_USE_CUDA)
  set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
  find_package(CUDA REQUIRED)
  if(ON)
    # Realm is providing entry points for the CUDA runtime API, so
    # filter anything referring to *cudart* out of the CUDA_LIBRARIES
    #  making sure our changes are visible to the scope above
    list(FILTER CUDA_LIBRARIES EXCLUDE REGEX cudart)
    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} PARENT_SCOPE)
  endif()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/LegionTargets.cmake)

if(Legion_USE_CUDA AND NOT Legion_CUDA_NVCC_FLAGS_SET)
  # CUDA has some issues propogating target interface properties so we
  # have to extract them from the target and manually add them to the nvcc
  # flags
  set(PROP $<TARGET_PROPERTY:Legion::Legion,INTERFACE_COMPILE_OPTIONS>)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${PROP}")
  set(PROP $<TARGET_PROPERTY:Legion::Legion,INTERFACE_COMPILE_DEFINITIONS>)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} $<$<BOOL:${PROP}>:-D$<JOIN:${PROP}, -D>>")
  set(PROP $<TARGET_PROPERTY:Legion::Legion,INTERFACE_INCLUDE_DIRECTORIES>)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} $<$<BOOL:${PROP}>:-I$<JOIN:${PROP}, -I>>")

  # Make sure we don't get duplicates
  set(Legion_CUDA_NVCC_FLAGS_SET ON)
endif()

# HIP is only a private dependency of Legion but has a few usage requirements
# so we add it in here
set(Legion_USE_HIP OFF)
if(Legion_USE_HIP)
  set(Legion_HIP_TARGET ROCM)
  find_package(HIP REQUIRED)
endif()

if(Legion_USE_HIP AND Legion_HIP_TARGET STREQUAL "CUDA" AND NOT Legion_HIP_HIPCC_FLAGS_SET)
  # CUDA has some issues propogating target interface properties so we
  # have to extract them from the target and manually add them to the nvcc
  # flags
  set(PROP $<TARGET_PROPERTY:Legion::Legion,INTERFACE_COMPILE_OPTIONS>)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${PROP}")
  set(PROP $<TARGET_PROPERTY:Legion::Legion,INTERFACE_COMPILE_DEFINITIONS>)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} $<$<BOOL:${PROP}>:-D$<JOIN:${PROP}, -D>>")
  set(PROP $<TARGET_PROPERTY:Legion::Legion,INTERFACE_INCLUDE_DIRECTORIES>)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} $<$<BOOL:${PROP}>:-I$<JOIN:${PROP}, -I>>")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ")

  # Make sure we don't get duplicates
  set(Legion_HIP_HIPCC_FLAGS_SET ON)
endif()

if(Legion_USE_HIP AND Legion_HIP_TARGET STREQUAL "ROCM" AND NOT Legion_HIP_HIPCC_FLAGS_SET)
  # flags
  set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS}  ")

  # Make sure we don't get duplicates
  set(Legion_HIP_HIPCC_FLAGS_SET ON)
endif()

# HWLOC is a private dependency and only needs to be pulled in for static
# builds
set(Legion_USE_HWLOC OFF)
if((NOT OFF) AND Legion_USE_HWLOC)
  set(HWLOC_INCLUDE_DIR )
  set(HWLOC_LIBRARY )
  find_package(HWLOC REQUIRED)
endif()

# ZLIB is a private dependency and only needs to be pulled in for static
# builds
set(Legion_USE_ZLIB ON)
if((NOT OFF) AND Legion_USE_ZLIB)
  set(ZLIB_INCLUDE_DIRS /usr/include)
  set(ZLIB_LIBRARIES /usr/lib/x86_64-linux-gnu/libz.so)
  find_package(ZLIB REQUIRED)
endif()

# OpenMP is an internal dependency, only needed so that users can key off of it
set(Legion_USE_OpenMP OFF)

# Python is an internal dependency, only needed so that users can key off of it
set(Legion_USE_Python ON)

# bring in Kokkos if needed
set(Legion_USE_Kokkos OFF)
if(Legion_USE_Kokkos)
  find_package(Kokkos REQUIRED OPTIONAL_COMPONENTS separable_compilation)

  # in order to build using Kokkos' exported compile options, we need to use
  #  the same compiler - newer versions of Kokkos will tell us, but for older
  #  versions, we need it from the configuration or the environment
  if(Kokkos_CXX_COMPILER)
    set(KOKKOS_CXX_COMPILER ${Kokkos_CXX_COMPILER})
  elseif(DEFINED ENV{KOKKOS_CXX_COMPILER})
    set(KOKKOS_CXX_COMPILER $ENV{KOKKOS_CXX_COMPILER})
  endif()
  if(NOT KOKKOS_CXX_COMPILER)
    message(FATAL_ERROR "to build correctly with Kokkos, the exact compiler used in the Kokkos build (typically set via CXX=... or -DCMAKE_CXX_COMPILER=...) must be provided in KOKKOS_CXX_COMPILER (either on the command line or from the environment)")
  endif()
endif()

# task registration across nodes often relies on being able to map function
#  pointers back to symbol names, so ask cmake to export symbols in binaries
set(CMAKE_ENABLE_EXPORTS ON)
