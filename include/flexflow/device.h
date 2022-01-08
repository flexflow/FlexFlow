#ifndef _FLEXFLOW_DEVICE_H_
#define _FLEXFLOW_DEVICE_H_

#if defined (FF_USE_CUDA) || defined (FF_USE_HIP_CUDA)
#include <cuda_runtime.h>
#elif defined (FF_USE_HIP_ROCM)
#include <hip/hip_runtime.h>
#else
#error "Unknown device"
#endif

namespace FlexFlow {
#if defined (FF_USE_CUDA) || defined (FF_USE_HIP_CUDA)
typedef cudaStream_t ffStream_t;
cudaError_t get_legion_stream(cudaStream_t *stream);
#elif defined (FF_USE_HIP_ROCM)
typedef hipStream_t ffStream_t;
hipError_t get_legion_stream(hipStream_t *stream);
#else
#error "Unknown device"
#endif
}; // namespace FlexFlow

#endif // _FLEXFLOW_DEVICE_H_