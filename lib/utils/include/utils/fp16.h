#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_FP16
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_FP16

#if defined(FF_USE_CUDA)
#include <cuda_fp16.h>
#elif defined(FF_USE_HIP_CUDA)
#include <cuda_fp16.h>
#elif defined(FF_USE_HIP_ROCM)
#include <hip/hip_fp16.h>
#endif

#endif
