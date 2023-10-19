#ifndef _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"

namespace FlexFlow {
namespace Kernels {
namespace Combine {

void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    DataType dataType);

void backward_kernel(ffStream_t stream,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const &input_grad,
                     DataType dataType);

} // namespace Combine
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_H
