#ifndef _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_H

#include "device.h"
#include "kernels/accessor.h"

namespace FlexFlow::Kernels::Combine {

void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow::Kernels::Combine

#endif // _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_H
