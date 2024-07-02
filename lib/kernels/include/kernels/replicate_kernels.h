#ifndef _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_H

#include "device.h"
#include "kernels/accessor.h"

namespace FlexFlow {
namespace Kernels {
namespace Replicate {

void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     GenericTensorAccessorW const &input,
                     GenericTensorAccessorR const &output,
                     size_t num_replicas);

} // namespace Replicate
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_H
