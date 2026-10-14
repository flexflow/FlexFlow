#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REDUCE_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_REDUCE_KERNELS_GPU_H

#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "kernels/reduce_per_device_state.dtg.h"
#include "op-attrs/operator_type.dtg.h"
#include "op-attrs/ops/reduce_op.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow::Kernels::Reduce {

ReducePerDeviceState gpu_init_kernel(PerDeviceFFHandle const &,
                                     ReduceOp const &reduce_op,
                                     size_t const &,
                                     TensorShape const &input_shape,
                                     TensorShape const &output_shape);

void gpu_forward_kernel(ffStream_t stream,
                        ReducePerDeviceState const &m,
                        float const *input_ptr,
                        float *output_ptr);

void gpu_backward_kernel(ffStream_t stream,
                         ReducePerDeviceState const &m,
                         float const *output_grad_ptr,
                         float *input_grad_ptr);

} // namespace FlexFlow::Kernels::Reduce

#endif
