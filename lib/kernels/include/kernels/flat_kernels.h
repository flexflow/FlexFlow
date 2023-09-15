#ifndef _FLEXFLOW_OPS_KERNELS_FLAT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_FLAT_KERNELS_H

#include "kernels/device.h"
#include "kernels/accessor.h"

namespace FlexFlow {
namespace Kernels {
namespace Flat {

void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorR input,
                    float *output_ptr);
void backward_kernel(ffStream_t stream,
                     GenericTensorAccessorR input,
                     float *input_grad_ptr,
                     float const *output_grad_ptr);

} // namespace Flat
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_FLAT_KERNELS_H
