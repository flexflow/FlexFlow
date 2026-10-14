#ifndef _FLEXFLOW_OPS_KERNELS_FLAT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_FLAT_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"

namespace FlexFlow {

void flat_forward_kernel(device_stream_t const &stream,
                         GenericTensorAccessorR const &input,
                         GenericTensorAccessorW const &output);

void flat_backward_kernel(device_stream_t const &stream,
                          GenericTensorAccessorR const &output_grad,
                          GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_FLAT_KERNELS_H
