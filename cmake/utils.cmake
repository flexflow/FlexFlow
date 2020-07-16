set(known_gpu_archs "60,75")
function(remove_duplicate_args __string)
  if(${__string})
    set(__list ${${__string}})
    separate_arguments(__list)
    list(REMOVE_DUPLICATES __list)
    foreach(__e ${__list})
      set(__str "${__str} ${__e}")
    endforeach()
    set(${__string} ${__str} PARENT_SCOPE)
  endif()
endfunction()
function(detect_installed_gpus out_variable)
  if(NOT CUDA_gpu_detect_output)
    set(__cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)
    file(WRITE ${__cufile} ""
      "#include <cstdio>\n"
      "int main()\n"
      "{\n"
      "  int count = 0;\n"
      "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
      "  if (count == 0) return -1;\n"
      "  for (int device = 0; device < count; ++device)\n"
      "  {\n"
      "    cudaDeviceProp prop;\n"
      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
      "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
      "  }\n"
      "  return 0;\n"
      "}\n")
    execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "--run" "${__cufile}"
                    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                    RESULT_VARIABLE __nvcc_res OUTPUT_VARIABLE __nvcc_out
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(__nvcc_res EQUAL 0)
      message(STATUS "No result from nvcc so building for 2.0")
      string(REPLACE "2.1" "2.1(2.0)" __nvcc_out "${__nvcc_out}")
      set(CUDA_gpu_detect_output ${__nvcc_out} CACHE INTERNAL "Returned GPU architetures from detect_gpus tool" FORCE)
    endif()
  endif()
  if(NOT CUDA_gpu_detect_output)
    message(STATUS "Automatic GPU detection failed. Building for architectures: ${known_gpu_archs}.")
    set(${out_variable} ${known_gpu_archs} PARENT_SCOPE)
  else()
    remove_duplicate_args(CUDA_gpu_detect_output)
    #Strip leading and trailing whitespaces
    string(STRIP "${CUDA_gpu_detect_output}" CUDA_gpu_detect_output)
    #Replace spaces in between with commas so you go from "5.2 6.1" to "5.2,6.1"
    string(REGEX REPLACE " " "," CUDA_gpu_detect_output "${CUDA_gpu_detect_output}")
    message(${CUDA_gpu_detect_output})
    string(REPLACE "." "" CUDA_gpu_detect_output "${CUDA_gpu_detect_output}")
    message(${CUDA_gpu_detect_output})
    set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
    message(STATUS "Automatic GPU ARCH detection: ${CUDA_gpu_detect_output}")
  endif()
endfunction()
