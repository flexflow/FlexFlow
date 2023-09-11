#ifndef _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"

namespace FlexFlow {

struct ConcatPerDeviceState {
  req<ff_dim_t> legion_axis;
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(ConcatPerDeviceState, legion_axis);

namespace Kernels {
namespace Concat {

ConcatPerDeviceState init_kernel(ff_dim_t legion_axis);

void forward_kernel(ffStream_t stream,
                    ConcatPerDeviceState const &m,
                    GenericTensorAccessorW const &output,
                    std::vector<FlexFlow::GenericTensorAccessorR> const &inputs,
                    int num_inputs);

void backward_kernel(
    ffStream_t stream,
    ConcatPerDeviceState const &m,
    GenericTensorAccessorR const &output_grad,
    std::vector<FlexFlow::GenericTensorAccessorW> const &input_grads,
    int num_inputs);

} // namespace Concat
} // namespace Kernels
} // namespace FlexFlow

#endif