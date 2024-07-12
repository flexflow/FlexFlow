#ifndef _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_CPU_H
#define _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_CPU_H

#include "device.h"
#include "kernels/accessor.h"

namespace FlexFlow {
namespace Kernels {
namespace Replicate {
namespace CPU {

void forward_kernel(GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(GenericTensorAccessorW const &input,
                     GenericTensorAccessorR const &output,
                     size_t num_replicas);

} // namespace CPU
} // namespace Replicate
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_CPU_H
