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

public:
  float eps;
  float *mean_ptr;
  char op_name[MAX_OPNAME];
};

namespace Kernels {
namespace RMSNorm {
void forward_kernel_wrapper(RMSNormMeta const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output,
                            GenericTensorAccessorR const &weight);
namespace Internal {

void forward_kernel(float const *input_ptr,
                    float const *weight_ptr,
                    float *output_ptr,
                    ffStream_t stream);
} // namespace Internal
} // namespace RMSNorm
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_RMSNORM_KERNELS_H