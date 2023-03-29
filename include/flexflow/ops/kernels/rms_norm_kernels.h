#ifndef _FLEXFLOW_OPS_KERNELS_RMSNORM_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_RMSNORM_KERNELS_H

#include "flexflow/accessor.h"
#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

class RMSNorm;

class RMSNormMeta : public OpMeta {
public:
  RMSNormMeta(FFHandler handler, RMSNorm const *rms);
  #if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnReduceTensorDescriptor_t reduceDesc;
  #else
    miopenTensorDescriptor_t inputTensor, outputTensor;
    miopenReduceTensorDescriptor_t reduceDesc;
  #endif

public:
  float eps;
  float *mean_ptr;
  char op_name[MAX_OPNAME];
};

namespace Kernels {
namespace RMSNorm {
void forward_kernel_wrapper(RMSNormMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorR const &weight,
                            GenericTensorAccessorW const &output);
namespace Internal {

void forward_kernel(float const *input_ptr,
                    float const *weight_ptr,
                    float *output_ptr,
                    Legion::coord_t dim_size,
                    ffStream_t stream);
} // namespace Internal
} // namespace RMSNorm
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_RMSNORM_KERNELS_H