#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ops/conv_2d/conv_2d_input_shape.h"
#include "op-attrs/ops/conv_2d/conv_2d_parallel_input_shape.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

std::vector<IncomingTensorRole>
    get_conv2d_incoming_tensor_roles(Conv2DAttrs const &attrs) {
  std::vector<IncomingTensorRole> result = {
      IncomingTensorRole::INPUT,
      IncomingTensorRole::WEIGHT,
  };

  if (attrs.use_bias) {
    result.push_back(IncomingTensorRole::WEIGHT);
  }

  return result;
}

TensorShape get_kernel_shape(Conv2DAttrs const &attrs,
                             TensorShape const &raw_input_shape) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported
  Conv2DInputShape input = parse_input_shape(raw_input_shape);

  return TensorShape{
      TensorDims{FFOrdered<size_t>{
          size_t_from_int(attrs.out_channels),
          input.num_channels,
          size_t_from_int(attrs.kernel_h),
          size_t_from_int(attrs.kernel_w),
      }},
      input.datatype,
  };
}

TensorShape get_bias_shape(Conv2DAttrs const &attrs,
                           TensorShape const &raw_input_shape) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported
  Conv2DInputShape input = parse_input_shape(raw_input_shape);

  return TensorShape{
      TensorDims{
          FFOrdered<size_t>{size_t_from_int(attrs.out_channels)},
      },
      input.datatype,
  };
}

TensorShape get_output_shape(Conv2DAttrs const &attrs,
                             TensorShape const &raw_input_shape) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported
  Conv2DInputShape input = parse_input_shape(raw_input_shape);

  size_t out_height =
      (input.height + (2 * attrs.padding_h) - attrs.kernel_h) / attrs.stride_h +
      1;
  size_t out_width =
      (input.width + (2 * attrs.padding_w) - attrs.kernel_w) / attrs.stride_w +
      1;

  assert(attrs.out_channels > 0);

  return TensorShape{TensorDims{FFOrdered<size_t>{
                         input.num_samples,
                         size_t_from_int(attrs.out_channels),
                         out_height,
                         out_width,
                     }},
                     input.datatype};
}

ParallelTensorShape get_kernel_shape(Conv2DAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported

  Conv2DParallelInputShape parsed = parse_parallel_input_shape(input);

  TensorShape unpar = get_kernel_shape(attrs, get_reduced_shape(input));

  assert(parsed.height_dim.degree == 1);
  assert(parsed.width_dim.degree == 1);

  SumDegree sum_degree = SumDegree{1};
  DiscardCopyDegree discard_copy_degree =
      DiscardCopyDegree{parsed.sample_dim.degree * parsed.sum_reduction_degree};
  FFOrdered<int> shard_degrees = {
      parsed.discard_copy_reduction_degree,
      parsed.channel_dim.degree,
      1,
      1,
  };

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

ParallelTensorShape get_bias_shape(Conv2DAttrs const &attrs,
                                   ParallelTensorShape const &input) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported

  Conv2DParallelInputShape parsed = parse_parallel_input_shape(input);

  TensorShape unpar = get_bias_shape(attrs, get_reduced_shape(input));

  SumDegree sum_degree =
      SumDegree{parsed.sum_reduction_degree * parsed.channel_dim.degree};
  DiscardCopyDegree discard_copy_degree =
      DiscardCopyDegree{parsed.height_dim.degree * parsed.width_dim.degree *
                        parsed.sample_dim.degree};
  FFOrdered<int> shard_degrees = {
      parsed.discard_copy_reduction_degree,
  };

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

ParallelTensorShape get_output_shape(Conv2DAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported

  Conv2DParallelInputShape parsed = parse_parallel_input_shape(input);

  TensorShape unpar = get_output_shape(attrs, get_reduced_shape(input));

  assert(parsed.height_dim.degree == 1);
  assert(parsed.width_dim.degree == 1);

  SumDegree sum_degree =
      SumDegree{parsed.sum_reduction_degree * parsed.channel_dim.degree};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{1};
  FFOrdered<int> shard_degrees = {
      parsed.sample_dim.degree,
      parsed.discard_copy_reduction_degree,
      1,
      1,
  };

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

} // namespace FlexFlow
