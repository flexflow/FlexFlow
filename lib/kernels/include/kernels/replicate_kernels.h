#ifndef _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_H

#include "device.h"
#include "kernels/accessor.h"

namespace FlexFlow::Kernels::Replicate {

void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     GenericTensorAccessorR const &output,
                     GenericTensorAccessorW const &input,
                     size_t num_replicas);

} // namespace FlexFlow::Kernels::Replicate

#endif // _FLEXFLOW_OPS_KERNELS_REPLICATE_KERNELS_H
