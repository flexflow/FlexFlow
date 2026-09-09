#ifndef _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"

namespace FlexFlow {

void reshape_forward_kernel(device_stream_t const &stream,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output);

void reshape_backward_kernel(device_stream_t const &stream,
                             GenericTensorAccessorR const &output,
                             GenericTensorAccessorR const &output_grad,
                             GenericTensorAccessorR const &input,
                             GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H
