#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

tl::expected<Pool2DAttrs, std::string>
    make_adaptive_pool2d_attrs(TensorDims const &input_dims,
                               positive_int output_h,
                               positive_int output_w,
                               PoolOp pool_type,
                               std::optional<Activation> const &activation) {
  // AdaptivePool2D semantics pulled from
  // https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work/63603993

  if (get_num_dims(input_dims) != 4) {
    return tl::unexpected(
        fmt::format("make_adaptive_pool2d_attrs expected input tensor to "
                    "have 4 dims, but received dims {}",
                    input_dims));
  }

  positive_int num_samples = dim_at_idx(input_dims, relative_ff_dim_t{0});
  positive_int num_channels = dim_at_idx(input_dims, relative_ff_dim_t{1});
  positive_int input_h = dim_at_idx(input_dims, relative_ff_dim_t{2});
  positive_int input_w = dim_at_idx(input_dims, relative_ff_dim_t{3});

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

  /**
   * Note that for some reason the stack overflow post linked above states that
   * `kernel_size = ind - (outd-1)*stride`, but some simplification yields
   * `kernel_size` = `ind - (outd - 1)*stride`
   *               = `ind - (outd - 1) * (ind / outd)`
   *               = `ind - ind + (ind  /outd)`
   *               = `ind / outd`
   *               = `stride`
   */

  positive_int kernel_h = positive_int{input_h / output_h};
  positive_int kernel_w = positive_int{input_w / output_w};

  positive_int stride_h = kernel_h;
  positive_int stride_w = kernel_w;

  Pool2DAttrs attrs = Pool2DAttrs{
      /*kernel_h=*/kernel_h,
      /*kernel_w=*/kernel_w,
      /*stride_h=*/stride_h,
      /*stride_w=*/stride_w,
      /*padding_h=*/0_n,
      /*padding_w=*/0_n,
      /*pool_type=*/pool_type,
      /*activation=*/activation,
  };

  TensorShape expected_ouput_shape = TensorShape{
      TensorDims{FFOrdered<positive_int>{
          num_samples,
          num_channels,
          output_h,
          output_w,
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

static positive_int calculate_output_size(positive_int input_size,
                                          nonnegative_int padding_size,
                                          positive_int kernel_size,
                                          positive_int stride) {
  int input_size_raw = input_size.int_from_positive_int();
  int padding_raw = padding_size.unwrap_nonnegative();
  int kernel_size_raw = kernel_size.int_from_positive_int();
  int stride_raw = stride.int_from_positive_int();

  return positive_int{
      (input_size_raw + (2 * padding_raw) - kernel_size_raw) / stride_raw + 1};
}

tl::expected<TensorShape, std::string>
    get_output_shape(Pool2DAttrs const &attrs, TensorShape const &input_shape) {
  if (get_num_dims(input_shape.dims) != 4) {
    return tl::unexpected(
        fmt::format("get_output_shape for Pool2DAttrs expected input tensor to "
                    "have 4 dims, but received shape {}",
                    input_shape));
  }

  positive_int num_samples = dim_at_idx(input_shape.dims, relative_ff_dim_t{0});
  positive_int num_channels =
      dim_at_idx(input_shape.dims, relative_ff_dim_t{1});
  positive_int input_height =
      dim_at_idx(input_shape.dims, relative_ff_dim_t{2});
  positive_int input_width = dim_at_idx(input_shape.dims, relative_ff_dim_t{3});

  positive_int output_height =
      calculate_output_size(/*input_size=*/input_height,
                            /*padding_size=*/attrs.padding_h,
                            /*kernel_size=*/attrs.kernel_h,
                            /*stride_size=*/attrs.stride_h);
  positive_int output_width =
      calculate_output_size(/*input_size=*/input_width,
                            /*padding_size=*/attrs.padding_w,
                            /*kernel_size=*/attrs.kernel_w,
                            /*stride_size=*/attrs.stride_w);

  return TensorShape{TensorDims{FFOrdered<positive_int>{
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

  return lift_shape_to_parallel_with_degrees(unpar, degrees);
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
