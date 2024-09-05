#include "op-attrs/ops/broadcast.h"
#include "op-attrs/tensor_dims.h"
#include "utils/record_formatter.h"

namespace FlexFlow {

RecordFormatter as_dot(BroadcastAttrs const &attrs) {
  RecordFormatter r;

  auto kv = [](std::string const &label, auto const &val) {
    RecordFormatter rr;
    rr << label << fmt::to_string(val);
    return rr;
  };

  for (int i = 0; i < num_dims(attrs.target_dims); i++) {
    r << kv(fmt::format("target_dims[{}]", i), dim_at_idx(attrs.target_dims, ff_dim_t{i}));
  }

  return r;
}

tl::expected<TensorShape, std::string>
    get_output_shape(BroadcastAttrs const &attrs,
                     TensorShape const &input_shape) {
  if (num_dims(attrs.target_dims) < num_dims(input_shape.dims)) {
    return tl::unexpected(fmt::format(
        "get_output_shape for Broadcast expected num_dims(input_dims) <= "
        "num_dims(target_dims), but recieved input_shape {} with num dims "
        "greater than target_dims {}",
        input_shape,
        attrs.target_dims));
  }

  if (tensor_dims_is_broadcastable_to(input_shape.dims, attrs.target_dims)) {
    return TensorShape{attrs.target_dims, input_shape.data_type};
  } else {
    return tl::unexpected(fmt::format(
        "Input tensor shape {} is not broadcastable to target dims {}",
        input_shape,
        attrs.target_dims));
  }
}

ParallelTensorShape get_output_shape(BroadcastAttrs const &,
                                     ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
