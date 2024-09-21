#include "op-attrs/ops/concat.h"
#include "op-attrs/dim_ordered/enumerate.h"
#include "op-attrs/dim_ordered/ff_ordered_from_map.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/all_of.h"
#include "utils/containers/are_all_same.h"
#include "utils/containers/require_all_same1.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/fmt/map.h"

namespace FlexFlow {

tl::expected<TensorShape, std::string>
    get_output_shape(ConcatAttrs const &attrs,
                     std::vector<TensorShape> const &inputs) {
  auto get_non_axis_dims = [&](TensorShape const &s) {
    std::map<ff_dim_t, size_t> dim_sizes = enumerate(ff_ordered(s.dims));
    dim_sizes.erase(attrs.axis);
    return dim_sizes;
  };

  if (inputs.size() <= 1) {
    return tl::unexpected(fmt::format("get_output_shape for Concat expected 2 "
                                      "or more input, but receieved {}",
                                      inputs));
  }

  if (attrs.axis.value < 0) {
    return tl::unexpected(fmt::format("ConcatAttrs requires axis >= 0"));
  }

  if (!are_all_same(transform(
          inputs, [](TensorShape const &s) { return num_dims(s); }))) {
    return tl::unexpected(
        fmt::format("get_output_shape for Concat expected all inputs to have "
                    "the same number of dimensions, but receieved {}",
                    inputs));
  }

  std::map<ff_dim_t, size_t> non_axis_dims = ({
    tl::expected<std::map<ff_dim_t, size_t>, std::string> returned =
        require_all_same1(transform(inputs, get_non_axis_dims));
    if (!returned.has_value()) {
      return tl::unexpected(returned.error());
    }
    returned.value();
  });

  std::vector<size_t> axis_dim_sizes = transform(
      inputs, [&](TensorShape const &s) { return dim_at_idx(s, attrs.axis); });

  size_t output_axis_dim_size = sum(axis_dim_sizes);

  non_axis_dims.insert({attrs.axis, output_axis_dim_size});

  DataType datatype = ({
    tl::expected<DataType, std::string> returned = require_all_same1(
        transform(inputs, [](TensorShape const &s) { return s.data_type; }));
    if (!returned.has_value()) {
      return tl::unexpected(returned.error());
    }
    returned.value();
  });

  return TensorShape{
      TensorDims{
          ff_ordered_from_map(non_axis_dims),
      },
      datatype,
  };
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(ConcatAttrs const &attrs,
                     std::vector<ParallelTensorShape> const &inputs) {
  TensorShape unpar = ({
    tl::expected<TensorShape, std::string> returned =
        get_output_shape(attrs, transform(inputs, get_reduced_shape));
    if (!returned.has_value()) {
      return tl::unexpected(returned.error());
    }
    returned.value();
  });

  SumDegree sum_degree = ({
    tl::expected<int, std::string> returned =
        require_all_same1(transform(inputs, get_sum_degree));
    if (!returned.has_value()) {
      return tl::unexpected(returned.error());
    }
    SumDegree{returned.value()};
  });

  DiscardCopyDegree discard_copy_degree = ({
    tl::expected<int, std::string> returned =
        require_all_same1(transform(inputs, get_discard_copy_degree));
    if (!returned.has_value()) {
      return tl::unexpected(returned.error());
    }
    DiscardCopyDegree{returned.value()};
  });

  if (!all_of(inputs, [&](ParallelTensorShape const &s) {
        return shard_dim_at_idx(s, attrs.axis).degree == 1;
      })) {
    return tl::unexpected(fmt::format(
        "get_output_shape for Concat expected input tensors to have parallel "
        "degree 1 in the concat axis dimension, but received {}",
        inputs));
  }

  ParallelTensorDimDegrees degrees = ({
    tl::expected<ParallelTensorDimDegrees, std::string> returned =
        require_all_same1(transform(inputs, [](ParallelTensorShape const &s) {
          return get_parallel_degrees(s);
        }));
    if (!returned.has_value()) {
      return tl::unexpected(returned.error());
    }
    returned.value();
  });

  return lift_to_parallel_with_degrees(unpar, degrees);
}

} // namespace FlexFlow
