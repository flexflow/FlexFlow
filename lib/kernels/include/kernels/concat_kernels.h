#ifndef _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"

namespace FlexFlow {
namespace Kernels {
namespace Concat {

void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorW const &output,
                    std::vector<FlexFlow::GenericTensorAccessorR> const &inputs,
                    int num_inputs,
                    ff_dim_t legion_axis);

void backward_kernel(
    ffStream_t stream,
    GenericTensorAccessorR const &output_grad,
    std::vector<FlexFlow::GenericTensorAccessorW> const &input_grads,
    int num_inputs,
    ff_dim_t legion_axis);

} // namespace Concat
} // namespace Kernels
} // namespace FlexFlow

#endif
