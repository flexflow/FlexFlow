#include "op-attrs/ops/broadcast.h"
#include "op-attrs/num_tensor_dims_t.h"
#include "op-attrs/tensor_dims.h"
#include "utils/not_implemented.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

TensorShape broadcast_get_output_shape(BroadcastAttrs const &attrs,
                                       TensorShape const &input_shape) {
  ASSERT(get_num_dims(attrs.target_dims) >= get_num_dims(input_shape.dims),
         fmt::format(
             "get_output_shape for Broadcast expected num_dims(input_dims) <= "
             "num_dims(target_dims), but received input_shape {} with num dims "
             "greater than target_dims {}",
             input_shape,
             attrs.target_dims));

  ASSERT(tensor_dims_is_broadcastable_to(input_shape.dims, attrs.target_dims),
         fmt::format(
             "Input tensor shape {} is not broadcastable to target dims {}",
             input_shape,
             attrs.target_dims));

  return TensorShape{attrs.target_dims, input_shape.data_type};
}

ParallelTensorShape
    broadcast_get_output_parallel_shape(BroadcastAttrs const &,
                                        ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
