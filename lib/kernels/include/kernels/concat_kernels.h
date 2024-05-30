#ifndef _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H

#include "device.h"
#include "kernels/accessor.h"
#include "kernels/concat_kernels.h"

namespace FlexFlow {
namespace Kernels {
namespace Concat {

void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorW const &output,
                    std::vector<GenericTensorAccessorR> const &inputs,
                    ff_dim_t axis);

void backward_kernel(ffStream_t stream,
                     GenericTensorAccessorR const &output_grad,
                     std::vector<GenericTensorAccessorW> const &input_grads,
                     ff_dim_t axis);

} // namespace Concat
} // namespace Kernels
} // namespace FlexFlow

#endif
