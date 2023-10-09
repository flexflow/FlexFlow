#ifndef _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H

#include "kernels/device.h"

namespace FlexFlow {

struct LayerNormPerDeviceState {
  float *mean, *rstd, *ds, *db, *scale, *bias;
};

namespace Kernels {
namespace LayerNorm {

LayerNormPerDeviceState init_kernel(PerDeviceFFHandle handle,
                                    int64_t batch_size);

void forward_kernel(ffStream_t stream,
                    LayerNormPerDeviceState const &m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    GenericTensorAccessorW const &gamma,
                    GenericTensorAccessorW const &beta,
                    DataType data_type,
                    int64_t batch_size,
                    int64_t num_elements,
                    float eps);

void backward_kernel(ffStream_t stream,
                     LayerNormPerDeviceState const &m,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &input_grad,
                     GenericTensorAccessorR const &gamma,
                     GenericTensorAccessorW const &gamma_grad,
                     GenericTensorAccessorW const &beta_grad,
                     DataType data_type,
                     int64_t batch_size,
                     int64_t num_elements,
                     float eps);

} // namespace LayerNorm
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H
