#ifndef _FLEXFLOW_KERNELS_PROFILING_H
#define _FLEXFLOW_KERNELS_PROFILING_H

#include "utils/optional.h"
#include "kernels/device.h"
#include "kernels/cuda_helper.h"

namespace FlexFlow {

template <typename F, typename ...Ts>
optional<float> profiling_wrapper(F const &f, bool profiling, Ts const &...ts) {
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  f(stream, ts...); 
  if (profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    return elapsed;
    /* printf("MultiHeadAttention forward time = %.2fms\n", elapsed); */
  }
  return tl::nullopt;
}

}

#endif
