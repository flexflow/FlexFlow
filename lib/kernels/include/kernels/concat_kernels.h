#ifndef _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"
#include "op-attrs/ops/concat_attrs.dtg.h"

namespace FlexFlow {

void concat_forward_kernel(device_stream_t const &stream,
                           ConcatAttrs const &attrs,
                           std::vector<GenericTensorAccessorR> const &inputs,
                           GenericTensorAccessorW const &output);

void concat_backward_kernel(
    device_stream_t const &stream,
    ConcatAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    std::vector<GenericTensorAccessorR> const &inputs,
    std::vector<GenericTensorAccessorW> const &input_grads);

} // namespace FlexFlow

#endif
