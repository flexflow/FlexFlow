#ifndef _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"
#include "op-attrs/ops/split_attrs.dtg.h"

namespace FlexFlow {

void split_forward_kernel(device_stream_t const &stream,
                          SplitAttrs const &attrs,
                          GenericTensorAccessorR const &input,
                          std::vector<GenericTensorAccessorW> const &outputs);

void split_backward_kernel(
    device_stream_t const &stream,
    SplitAttrs const &attrs,
    std::vector<GenericTensorAccessorR> const &outputs,
    std::vector<GenericTensorAccessorR> const &output_grads,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_SPLIT_KERNELS_H
