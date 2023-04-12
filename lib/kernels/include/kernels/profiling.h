#ifndef _FLEXFLOW_KERNELS_PROFILING_H
#define _FLEXFLOW_KERNELS_PROFILING_H

#include "utils/optional.h"
#include "kernels/device.h"

namespace FlexFlow {

template <typename F, typename ...Ts>
optional<float> profiling_wrapper(F const &f, bool profiling, Ts &&...ts) { 
  ffStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  ffEvent_t t_start, t_end;
  if (profiling) {
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
#elif defined(FF_USE_HIP_ROCM)
    hipEventCreate(&t_start);
    hipEventCreate(&t_end);
    hipEventRecord(t_start, stream);
#endif
  }
  f(stream, ts...); 
  if (profiling) {
    float elapsed = 0;
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
#elif defined(FF_USE_HIP_ROCM)
    hipEventRecord(t_end, stream);
    checkCUDA(hipEventSynchronize(t_end));
    checkCUDA(hipEventElapsedTime(&elapsed, t_start, t_end));
    hipEventDestroy(t_start);
    hipEventDestroy(t_end);
#endif
    return elapsed;
    /* printf("MultiHeadAttention forward time = %.2fms\n", elapsed); */
  }
  return tl::nullopt;
}

}

#endif
