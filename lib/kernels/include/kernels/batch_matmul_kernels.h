#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_MATMUL_KERNELS_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_MATMUL_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "op-attrs/ops/batch_matmul_attrs.dtg.h"

namespace FlexFlow {

void batch_matmul_forward_kernel(device_stream_t const &stream,
                                 device_handle_t const &handle,
                                 GenericTensorAccessorR const &input_lhs,
                                 GenericTensorAccessorR const &input_rhs,
                                 GenericTensorAccessorW const &output);

void batch_matmul_backward_kernel(device_stream_t const &stream,
                                  device_handle_t const &handle,
                                  GenericTensorAccessorR const &output,
                                  GenericTensorAccessorR const &output_grad,
                                  GenericTensorAccessorR const &input_lhs,
                                  GenericTensorAccessorW const &input_lhs_grad,
                                  GenericTensorAccessorR const &input_rhs,
                                  GenericTensorAccessorW const &input_rhs_grad);

} // namespace FlexFlow

#endif
