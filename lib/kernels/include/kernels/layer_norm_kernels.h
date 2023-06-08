#ifndef _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H

#include "kernels/device.h"
#include "kernels/per_device_op_state.h"

namespace FlexFlow {

class LayerNormPerDeviceState : public PerDeviceOpState {
public:
  LayerNormPerDeviceState(FFHandler handle, bool elementwise_affine_,
                          int64_t effective_batch_size_,
                          int64_t effective_num_elements_, bool profiling_,
                          float eps_);

public:
  bool elementwise_affine;
  int64_t effective_batch_size, effective_num_elements;
  float eps;
  float *mean, *rstd, *ds, *db, *scale, *bias;
  char op_name[MAX_OPNAME];
  DataType data_type;
};

namespace Kernels {
namespace LayerNorm {

void forward_kernel(ffStream_t stream, LayerNormPerDeviceState const *m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    GenericTensorAccessorW const &gamma,
                    GenericTensorAccessorW const &beta);

void backward_kernel(ffStream_t stream, LayerNormPerDeviceState const *m,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &input_grad,
                     GenericTensorAccessorR const &gamma,
                     GenericTensorAccessorW const &gamma_grad,
                     GenericTensorAccessorW const &beta_grad);

} // namespace LayerNorm
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H
