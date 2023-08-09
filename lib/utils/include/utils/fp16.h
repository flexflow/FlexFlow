#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_FP16
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_FP16

#include "hash-utils.h"

#if defined(FF_USE_CUDA)
#include <cuda_fp16.h>
#elif defined(FF_USE_HIP_CUDA)
#include <cuda_fp16.h>
#elif defined(FF_USE_HIP_ROCM)
#include <hip/hip_fp16.h>
#else
static_assert(false, "Could not find half definition");
#endif

namespace std {

template <>
struct hash<::half> {
  size_t operator()(::half h) const;
};

} // namespace std

#endif
