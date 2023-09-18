#ifndef _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H

#include "accessor.h"
#include "kernels/device.h"

namespace FlexFlow {
namespace Kernels {
namespace Gather {
void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorR const &index,
                    GenericTensorAccessorW const &output,
                    size_t stride,
                    legion_dim_t dim);
void backward_kernel(ffStream_t stream,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &index,
                     GenericTensorAccessorW const &input_grad,
                     size_t stride,
                     legion_dim_t dim);
} // namespace Gather
} // namespace Kernels
} // namespace FlexFlow

#endif
