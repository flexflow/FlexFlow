#ifndef _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_CPU_H
#define _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_CPU_H

#include "device.h"
#include "kernels/accessor.h"

namespace FlexFlow {
namespace Kernels {
namespace Combine {
namespace CPU {

void forward_kernel(GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const &input_grad);

} // namespace CPU
} // namespace Combine
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_CPU_H
