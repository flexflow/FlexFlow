#include "device.h"

namespace FlexFlow {

char const *getCudaErrorString(cudaError_t status) {
  return cudaGetErrorString(status);
}

ffError_t ffEventCreate(ffEvent_t *e) {
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  return cudaEventCreate(e);
#elif defined(FF_USE_HIP_ROCM)
  return hipEventCreate(e);
#endif
}

ffError_t ffEventDestroy(ffEvent_t &e) {
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  return cudaEventDestroy(e);
#elif defined(FF_USE_HIP_ROCM)
  return hipEventDestroy(e);
#endif
}

ffError_t ffEventRecord(ffEvent_t &e, ffStream_t stream) {
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  return cudaEventRecord(e, stream);
#elif defined(FF_USE_HIP_ROCM)
  return hipEventRecord(e, stream);
#endif
}

ffError_t ffEventSynchronize(ffEvent_t &e) {
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  return cudaEventSynchronize(e);
#elif defined(FF_USE_HIP_ROCM)
  return hipEventSynchronize(e);
#endif
}

ffError_t
    ffEventElapsedTime(float *elapsed, ffEvent_t &start, ffEvent_t &stop) {
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  return cudaEventElapsedTime(elapsed, start, stop);
#elif defined(FF_USE_HIP_ROCM)
  return cudaEventElapsedTime(elapsed, start, stop);
#endif
}

} // namespace FlexFlow
