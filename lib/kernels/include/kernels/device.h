#ifndef _FLEXFLOW_DEVICE_H_
#define _FLEXFLOW_DEVICE_H_

#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include <cuda_runtime.h>
#include <cudnn.h>
#elif defined(FF_USE_HIP_ROCM)
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#else
#error "Unknown device"
#endif

namespace FlexFlow {
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
typedef cudaStream_t ffStream_t;
cudaError_t get_legion_stream(cudaStream_t *stream);
typedef cudnnTensorDescriptor_t ffTensorDescriptor_t;
typedef cudnnActivationDescriptor_t ffActivationDescriptor_t;
typedef cudnnPoolingDescriptor_t ffPoolingDescriptor_t;
#elif defined(FF_USE_HIP_ROCM)
typedef hipStream_t ffStream_t;
hipError_t get_legion_stream(hipStream_t *stream);
typedef miopenTensorDescriptor_t ffTensorDescriptor_t;
typedef miopenActivationDescriptor_t ffActivationDescriptor_t;
typedef miopenPoolingDescriptor_t ffPoolingDescriptor_t;
#else
#error "Unknown device"
#endif
}; // namespace FlexFlow

#endif // _FLEXFLOW_DEVICE_H_
