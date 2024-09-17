#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

tl::expected<Pool2DAttrs, std::string>
    make_adaptive_pool2d_attrs(TensorDims const &input_dims,
                               int output_h,
                               int output_w,
                               PoolOp pool_type,
                               std::optional<Activation> const &activation) {
  // AdaptivePool2D semantics pulled from
  // https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work/63603993

  if (num_dims(input_dims) != 4) {
    return tl::unexpected(
        fmt::format("make_adaptive_pool2d_attrs expected input tensor to "
                    "have 4 dims, but received dims {}",
                    input_dims));
  }

  size_t num_samples = dim_at_idx(input_dims, ff_dim_t{0});
  size_t num_channels = dim_at_idx(input_dims, ff_dim_t{1});
  size_t input_h = dim_at_idx(input_dims, ff_dim_t{2});
  size_t input_w = dim_at_idx(input_dims, ff_dim_t{3});

  if (input_h % output_h != 0) {
    return tl::unexpected(fmt::format(
        "Currently make_adaptive_pool2d_attrs only supports input_h % output_h "
        "== 0, but received input_h={} and output_h={} (input_dims={}). If you "
        "need input_h % output_h != 0 supported, please create an issue.",
        input_h,
        output_h,
        input_dims));
  }

  if (input_w % output_w != 0) {
    return tl::unexpected(fmt::format(
        "Currently make_adaptive_pool2d_attrs only supports input_w % output_w "
        "== 0, but received input_w={} and output_w={} (input_dims={}). If you "
        "need input_w % output_w != 0 supported, please create an issue.",
        input_w,
        output_w,
        input_dims));
  }

  // Note that for some reason the stack overflow post linked above states that 
  // `kernel_size = ind - (outd-1)*stride`, but some simplification yields
  // `kernel_size` = `ind - (outd - 1)*stride`
  //               = `ind - (outd - 1) * (ind / outd)` 
  //               = `ind - ind + (ind  /outd)` 
  //               = `ind / outd` 
  //               = `stride`

  int kernel_h = input_h / output_h;
  int kernel_w = input_w / output_w;

  int stride_h = kernel_h;
  int stride_w = kernel_w;

  Pool2DAttrs attrs = Pool2DAttrs{
      /*kernel_h=*/kernel_h,
      /*kernel_w=*/kernel_w,
      /*stride_h=*/stride_h,
      /*stride_w=*/stride_w,
      /*padding_h=*/0,
      /*padding_w=*/0,
      /*pool_type=*/pool_type,
      /*activation=*/activation,
  };

  TensorShape expected_ouput_shape = TensorShape{
      TensorDims{FFOrdered<size_t>{
          num_samples,
          num_channels,
          size_t_from_int(output_h),
          size_t_from_int(output_w),
      }},
      DataType::FLOAT,
  };

  TensorShape output_shape = ({
    tl::expected<TensorShape, std::string> result =
        get_output_shape(attrs, TensorShape{input_dims, DataType::FLOAT});
    if (!result.has_value()) {
      return tl::unexpected(result.error());
    }
    result.value();
  });

  if (output_shape != expected_ouput_shape) {
    return tl::unexpected(
        fmt::format("Result of make_adaptive_pool_2d (i.e., {}) should produce "
                    "expected output shape {}, but produced {}. This is a bug "
                    "in FlexFlow, Please create an issue.",
                    attrs,
                    expected_ouput_shape,
                    output_shape));
  }

  return attrs;
}

tl::expected<TensorShape, std::string>
    get_output_shape(Pool2DAttrs const &attrs, TensorShape const &input_shape) {
  if (num_dims(input_shape) != 4) {
    return tl::unexpected(
        fmt::format("get_output_shape for Pool2DAttrs expected input tensor to "
                    "have 4 dims, but received shape {}",
                    input_shape));
  }

  size_t num_samples = dim_at_idx(input_shape, ff_dim_t{0});
  size_t num_channels = dim_at_idx(input_shape, ff_dim_t{1});
  size_t input_height = dim_at_idx(input_shape, ff_dim_t{2});
  size_t input_width = dim_at_idx(input_shape, ff_dim_t{3});

  size_t output_height =
      (input_height + 2 * attrs.padding_h - attrs.kernel_h) / attrs.stride_h +
      1;

  size_t output_width =
      (input_width + 2 * attrs.padding_w - attrs.kernel_w) / attrs.stride_w + 1;

  return TensorShape{TensorDims{FFOrdered<size_t>{
                         num_samples,
                         num_channels,
                         output_height,
                         output_width,
                     }},
                     input_shape.data_type};
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(Pool2DAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {
  TensorShape unpar = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_output_shape(attrs, get_reduced_shape(input_shape));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }
    result_unpar.value();
  });

  ParallelTensorDimDegrees degrees = ({
    tl::expected<ParallelTensorDimDegrees, std::string> result_degrees =
        get_output_parallel_dim_degrees(attrs,
                                        get_parallel_degrees(input_shape));
    if (!result_degrees.has_value()) {
      return tl::unexpected(result_degrees.error());
    }
    result_degrees.value();
  });

  return lift_to_parallel_with_degrees(unpar, degrees);
}

tl::expected<ParallelTensorDimDegrees, std::string>
    get_output_parallel_dim_degrees(
        Pool2DAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  if (input_degrees.sum_degree.value > 1) {
    if (attrs.pool_type == PoolOp::MAX) {
      return tl::unexpected(fmt::format(
          "get_output_parallel_dim_degrees for Pool2DAttrs with PoolOp::MAX "
          "expected input sum degree == 1, but received {}",
          input_degrees));
    } else if (attrs.activation.has_value()) {
      return tl::unexpected(fmt::format(
          "get_output_parallel_dim_degrees for Pool2DAttrs with activation={} "
          "expected input sum degree == 1, but received {}",
          attrs.activation.value(),
          input_degrees));
    }
  }

  return input_degrees;
}

} // namespace FlexFlow
