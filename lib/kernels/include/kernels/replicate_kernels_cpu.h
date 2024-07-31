#ifndef _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_CPU_H
#define _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_CPU_H

#include "device.h"
#include "kernels/accessor.h"

namespace FlexFlow {
namespace Kernels {
namespace Replicate {

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output);

void cpu_backward_kernel(GenericTensorAccessorW const &input,
                         GenericTensorAccessorR const &output,
                         size_t num_replicas);

} // namespace Replicate
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_CPU_H
