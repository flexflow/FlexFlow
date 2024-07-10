#ifndef _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_LAYER_NORM_KERNELS_H

#include "device.h"
#include "kernels/allocation.h"
#include "kernels/ff_handle.h"

namespace FlexFlow {

struct LayerNormPerDeviceState {
  PerDeviceFFHandle handle;
  bool elementwise_affine;
  int64_t effective_batch_size, effective_num_elements;
  float eps;
  float *mean, *rstd, *ds, *db, *scale, *bias;
  DataType data_type;
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(LayerNormPerDeviceState,
                                             handle,
                                             elementwise_affine,
                                             effective_batch_size,
                                             effective_num_elements,
                                             eps,
                                             mean,
                                             rstd,
                                             ds,
                                             db,
                                             scale,
                                             bias,
                                             data_type);

namespace Kernels {
namespace LayerNorm {

// todo: this may have some problem.
LayerNormPerDeviceState init_kernel(PerDeviceFFHandle const &handle,
                                    Allocator &allocator,
                                    bool elementwise_affine,
                                    int64_t effective_batch_size,
                                    int64_t effective_num_elements,
                                    float eps);

void forward_kernel(ffStream_t stream,
                    LayerNormPerDeviceState const &m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    GenericTensorAccessorW const &gamma,
                    GenericTensorAccessorW const &beta);

void backward_kernel(ffStream_t stream,
                     LayerNormPerDeviceState const &m,
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
