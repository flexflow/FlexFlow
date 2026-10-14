#ifndef _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H

#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/ff_handle.h"
#include "kernels/reduce_per_device_state.dtg.h"
#include "op-attrs/operator_type.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "pcg/device_type.dtg.h"

namespace FlexFlow::Kernels::Reduce {

std::optional<ReducePerDeviceState>
    init_kernel(DeviceType device_type,
                device_handle_t const &handle,
                ReduceOp const &reduce_op,
                size_t const &reduction_size,
                TensorShape const &input_shape,
                TensorShape const &output_shape);

void forward_kernel(device_stream_t const &stream,
                    std::optional<ReducePerDeviceState> const &per_device_state,
                    float const *input_ptr,
                    float *output_ptr);

void backward_kernel(
    device_stream_t const &stream,
    std::optional<ReducePerDeviceState> const &per_device_state,
    float const *output_grad_ptr,
    float *input_grad_ptr);

} // namespace FlexFlow::Kernels::Reduce

#endif // _FLEXFLOW_OPS_KERNELS_REDUCE_KERNELS_H
