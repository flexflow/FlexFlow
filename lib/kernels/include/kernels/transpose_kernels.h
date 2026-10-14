#ifndef _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"
#include "op-attrs/ops/transpose_attrs.dtg.h"

namespace FlexFlow {

void transpose_forward_kernel(device_stream_t const &stream,
                              TransposeAttrs const &attrs,
                              GenericTensorAccessorR const &input,
                              GenericTensorAccessorW const &output);

void transpose_backward_kernel(device_stream_t const &stream,
                               TransposeAttrs const &attrs,
                               GenericTensorAccessorR const &out_grad,
                               GenericTensorAccessorW const &in_grad);

} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H
