#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/initializers/kaiming_initializer_mode.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/tensor_dims.h"
#include "utils/fmt/optional.h"
#include "utils/integer_conversions.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

std::map<TensorSlotName, IncomingTensorRole>
    get_conv2d_incoming_tensor_roles(Conv2DAttrs const &attrs) {
  std::map<TensorSlotName, IncomingTensorRole> result = {
      {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
      {TensorSlotName::FILTER, IncomingTensorRole::WEIGHT},
  };

  if (attrs.use_bias) {
    result[TensorSlotName::BIAS] = IncomingTensorRole::WEIGHT;
  }

  return result;
}

TensorShape conv2d_get_kernel_shape(Conv2DAttrs const &attrs,
                                    TensorShape const &raw_input_shape) {
  ASSERT(get_num_dims(raw_input_shape.dims) == 4);

  positive_int input_n = dim_at_idx(raw_input_shape.dims, ff_dim_t{0_n});
  positive_int input_c = dim_at_idx(raw_input_shape.dims, ff_dim_t{1_n});
  positive_int input_h = dim_at_idx(raw_input_shape.dims, ff_dim_t{2_n});
  positive_int input_w = dim_at_idx(raw_input_shape.dims, ff_dim_t{3_n});

  return TensorShape{
      TensorDims{FFOrdered<positive_int>{
          attrs.out_channels,
          positive_int{input_c / attrs.groups},
          attrs.kernel_h,
          attrs.kernel_w,
      }},
      raw_input_shape.data_type,
  };
}

TensorShape conv2d_get_bias_shape(Conv2DAttrs const &attrs,
                                  TensorShape const &raw_input_shape) {
  ASSERT(get_num_dims(raw_input_shape.dims) == 4);

  positive_int input_n = dim_at_idx(raw_input_shape.dims, ff_dim_t{0_n});
  positive_int input_c = dim_at_idx(raw_input_shape.dims, ff_dim_t{1_n});
  positive_int input_h = dim_at_idx(raw_input_shape.dims, ff_dim_t{2_n});
  positive_int input_w = dim_at_idx(raw_input_shape.dims, ff_dim_t{3_n});

  return TensorShape{
      TensorDims{
          FFOrdered<positive_int>{attrs.out_channels},
      },
      raw_input_shape.data_type,
  };
}

TensorShape conv2d_get_output_shape(Conv2DAttrs const &attrs,
                                    TensorShape const &raw_input_shape) {

  positive_int input_n = dim_at_idx(raw_input_shape.dims, ff_dim_t{0_n});
  positive_int input_c = dim_at_idx(raw_input_shape.dims, ff_dim_t{1_n});
  positive_int input_h = dim_at_idx(raw_input_shape.dims, ff_dim_t{2_n});
  positive_int input_w = dim_at_idx(raw_input_shape.dims, ff_dim_t{3_n});

  auto calculate_output_dim_size = [](positive_int input_size,
                                      nonnegative_int padding_size,
                                      positive_int kernel_size,
                                      positive_int stride) -> positive_int {
    int input_size_raw = input_size.int_from_positive_int();
    int padding_raw = padding_size.unwrap_nonnegative();
    int kernel_size_raw = kernel_size.int_from_positive_int();
    int stride_raw = stride.int_from_positive_int();

    return positive_int{
        (input_size_raw + (2 * padding_raw) - kernel_size_raw) / stride_raw + 1,
    };
  };

  positive_int out_height =
      calculate_output_dim_size(/*input_size=*/input_h,
                                /*padding_size=*/attrs.padding_h,
                                /*kernel_size=*/attrs.kernel_h,
                                /*stride_size=*/attrs.stride_h);
  positive_int out_width =
      calculate_output_dim_size(/*input_size=*/input_w,
                                /*padding_size=*/attrs.padding_w,
                                /*kernel_size=*/attrs.kernel_w,
                                /*stride_size=*/attrs.stride_w);

  return TensorShape{
      TensorDims{
          FFOrdered<positive_int>{
              input_n,
              attrs.out_channels,
              out_height,
              out_width,
          },
      },
      raw_input_shape.data_type,
  };
}

std::map<TensorSlotName, TensorShape>
    conv2d_get_weight_shapes(Conv2DAttrs const &attrs,
                             TensorShape const &input_shape) {
  std::map<TensorSlotName, TensorShape> weight_shapes = {
      {
          TensorSlotName::FILTER,
          conv2d_get_kernel_shape(attrs, input_shape),
      },
  };

  if (attrs.use_bias) {
    weight_shapes.insert({
        TensorSlotName::BIAS,
        conv2d_get_bias_shape(attrs, input_shape),
    });
  }

  return weight_shapes;
}

ParallelTensorDimDegrees conv2d_get_kernel_parallel_dim_degrees(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input_dim_degrees) {
  ASSERT(get_ptensor_dim_degrees_num_shard_dims(input_dim_degrees) ==
         num_ptensor_shard_dims_t{4_n});

  positive_int input_sum_degree = input_dim_degrees.sum_degree.value;
  positive_int input_discard_copy_degree =
      input_dim_degrees.discard_copy_degree.value;
  positive_int input_n_degree =
      input_dim_degrees.shard_degrees.at(ff_dim_t{0_n});
  positive_int input_c_degree =
      input_dim_degrees.shard_degrees.at(ff_dim_t{1_n});
  ASSERT(input_dim_degrees.shard_degrees.at(ff_dim_t{2_n}) == 1);
  ASSERT(input_dim_degrees.shard_degrees.at(ff_dim_t{3_n}) == 1);

  SumDegree sum_degree = SumDegree{1_p};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      input_n_degree * input_sum_degree,
  };

  if (input_c_degree % attrs.groups == 0) {
    FFOrdered<positive_int> shard_degrees = FFOrdered{
        input_discard_copy_degree * attrs.groups,
        positive_int{input_c_degree / attrs.groups},
        1_p,
        1_p,
    };

    return ParallelTensorDimDegrees{
        /*sum_degree=*/sum_degree,
        /*discard_copy_degree=*/discard_copy_degree,
        /*shard_degrees=*/shard_degrees,
    };
  } else if (attrs.groups % input_c_degree == 0) {
    FFOrdered<positive_int> shard_degrees = FFOrdered{
        input_discard_copy_degree * input_c_degree,
        1_p,
        1_p,
        1_p,
    };

    return ParallelTensorDimDegrees{
        /*sum_degree=*/sum_degree,
        /*discard_copy_degree=*/discard_copy_degree,
        /*shard_degrees=*/shard_degrees,
    };
  } else {
    PANIC("input_channel_degree and group count are not compatible",
          input_c_degree,
          attrs.groups);
  }
}

ParallelTensorDimDegrees conv2d_get_bias_parallel_dim_degrees(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input_dim_degrees) {
  ASSERT(get_ptensor_dim_degrees_num_shard_dims(input_dim_degrees) ==
         num_ptensor_shard_dims_t{4_n});

  positive_int input_sum_degree = input_dim_degrees.sum_degree.value;
  positive_int input_discard_copy_degree =
      input_dim_degrees.discard_copy_degree.value;
  positive_int input_n_degree =
      input_dim_degrees.shard_degrees.at(ff_dim_t{0_n});
  positive_int input_c_degree =
      input_dim_degrees.shard_degrees.at(ff_dim_t{1_n});
  ASSERT(input_dim_degrees.shard_degrees.at(ff_dim_t{2_n}) == 1);
  ASSERT(input_dim_degrees.shard_degrees.at(ff_dim_t{3_n}) == 1);

  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{input_n_degree};

  if (input_c_degree % attrs.groups == 0) {
    SumDegree sum_degree = SumDegree{
        input_sum_degree * positive_int{input_c_degree / attrs.groups},
    };

    FFOrdered<positive_int> shard_degrees = FFOrdered{
        input_discard_copy_degree * attrs.groups,
    };

    return ParallelTensorDimDegrees{
        /*sum_degree=*/sum_degree,
        /*discard_copy_degree=*/discard_copy_degree,
        /*shard_degrees=*/shard_degrees,
    };
  } else if (attrs.groups % input_c_degree == 0) {
    SumDegree sum_degree = SumDegree{
        input_sum_degree,
    };

    FFOrdered<positive_int> shard_degrees = FFOrdered{
        input_discard_copy_degree * input_c_degree,
    };

    return ParallelTensorDimDegrees{
        /*sum_degree=*/sum_degree,
        /*discard_copy_degree=*/discard_copy_degree,
        /*shard_degrees=*/shard_degrees,
    };
  } else {
    PANIC("input_channel_degree and group count are not compatible",
          input_c_degree,
          attrs.groups);
  }
}

ParallelTensorDimDegrees conv2d_get_output_parallel_dim_degrees(
    Conv2DAttrs const &attrs,
    ParallelTensorDimDegrees const &input_dim_degrees) {
  ASSERT(get_ptensor_dim_degrees_num_shard_dims(input_dim_degrees) ==
         num_ptensor_shard_dims_t{4_n});

  positive_int input_sum_degree = input_dim_degrees.sum_degree.value;
  positive_int input_discard_copy_degree =
      input_dim_degrees.discard_copy_degree.value;
  positive_int input_n_degree =
      input_dim_degrees.shard_degrees.at(ff_dim_t{0_n});
  positive_int input_c_degree =
      input_dim_degrees.shard_degrees.at(ff_dim_t{1_n});
  ASSERT(input_dim_degrees.shard_degrees.at(ff_dim_t{2_n}) == 1);
  ASSERT(input_dim_degrees.shard_degrees.at(ff_dim_t{3_n}) == 1);

  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{1_p};

  if (input_c_degree % attrs.groups == 0) {
    SumDegree sum_degree = SumDegree{
        input_sum_degree * positive_int{input_c_degree / attrs.groups},
    };

    FFOrdered<positive_int> shard_degrees = FFOrdered{
        input_n_degree,
        input_discard_copy_degree * attrs.groups,
        1_p,
        1_p,
    };

    return ParallelTensorDimDegrees{
        /*sum_degree=*/sum_degree,
        /*discard_copy_degree=*/discard_copy_degree,
        /*shard_degrees=*/shard_degrees,
    };
  } else if (attrs.groups % input_c_degree == 0) {
    SumDegree sum_degree = SumDegree{
        input_sum_degree,
    };

    FFOrdered<positive_int> shard_degrees = FFOrdered{
        input_n_degree,
        input_discard_copy_degree * input_c_degree,
        1_p,
        1_p,
    };

    return ParallelTensorDimDegrees{
        /*sum_degree=*/sum_degree,
        /*discard_copy_degree=*/discard_copy_degree,
        /*shard_degrees=*/shard_degrees,
    };
  } else {
    PANIC("input_channel_degree and group count are not compatible",
          input_c_degree,
          attrs.groups);
  }
}

ParallelTensorShape
    conv2d_get_kernel_parallel_shape(Conv2DAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  TensorShape unpar = conv2d_get_kernel_shape(attrs, get_reduced_shape(input));
  ParallelTensorDimDegrees degrees = conv2d_get_kernel_parallel_dim_degrees(
      attrs, get_parallel_degrees(input));

  return lift_shape_to_parallel_with_degrees(unpar, degrees);
}

ParallelTensorShape
    conv2d_get_bias_parallel_shape(Conv2DAttrs const &attrs,
                                   ParallelTensorShape const &input) {
  TensorShape unpar = conv2d_get_bias_shape(attrs, get_reduced_shape(input));
  ParallelTensorDimDegrees degrees =
      conv2d_get_bias_parallel_dim_degrees(attrs, get_parallel_degrees(input));

  return lift_shape_to_parallel_with_degrees(unpar, degrees);
}

ParallelTensorShape
    conv2d_get_output_parallel_shape(Conv2DAttrs const &attrs,
                                     ParallelTensorShape const &input) {

  TensorShape unpar = conv2d_get_output_shape(attrs, get_reduced_shape(input));
  ParallelTensorDimDegrees degrees = conv2d_get_output_parallel_dim_degrees(
      attrs, get_parallel_degrees(input));

  return lift_shape_to_parallel_with_degrees(unpar, degrees);
}

std::map<TensorSlotName, ParallelTensorShape>
    conv2d_get_weight_parallel_shapes(Conv2DAttrs const &attrs,
                                      ParallelTensorShape const &input_shape) {
  std::map<TensorSlotName, ParallelTensorShape> weight_shapes = {
      {
          TensorSlotName::FILTER,
          conv2d_get_kernel_parallel_shape(attrs, input_shape),
      },
  };

  if (attrs.use_bias) {
    weight_shapes.insert({
        TensorSlotName::BIAS,
        conv2d_get_bias_parallel_shape(attrs, input_shape),
    });
  }

  return weight_shapes;
}

/**
 * @brief Chosen to match pytorch implementation
 *
 * see
 * https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/nn/modules/conv.py#L178-L187
 */
std::map<TensorSlotName, InitializerAttrs> conv2d_get_initializers(
    Conv2DAttrs const &attrs,
    TensorShape const &input_shape,
    std::optional<InitializerAttrs> maybe_kernel_initializer,
    std::optional<InitializerAttrs> maybe_bias_initializer) {

  TensorShape kernel_shape = conv2d_get_kernel_shape(attrs, input_shape);

  InitializerAttrs kernel_default_initializer =
      InitializerAttrs{KaimingNormalAttrs{
          /*a=*/sqrtf(5.0),
          /*mode=*/KaimingInitializerMode::FAN_IN,
          /*nonlinearity=*/KaimingInitializerNonlinearity::LEAKY_RELU,
          /*seed=*/0,
      }};

  InitializerAttrs kernel_initializer =
      maybe_kernel_initializer.value_or(kernel_default_initializer);

  positive_int fan_in =
      calculate_fan_for_mode(kernel_shape.dims, KaimingInitializerMode::FAN_IN);

  float bound = 1 / sqrtf(static_cast<float>(fan_in.int_from_positive_int()));

  InitializerAttrs bias_default_initializer =
      InitializerAttrs{UniformInitializerAttrs{
          /*seed=*/0,
          /*min_val=*/-bound,
          /*max_val=*/bound,
      }};

  InitializerAttrs bias_initializer =
      maybe_bias_initializer.value_or(bias_default_initializer);

  if (attrs.use_bias) {
    return {
        {TensorSlotName::FILTER, kernel_initializer},
        {TensorSlotName::BIAS, bias_initializer},
    };
  } else {
    return {
        {TensorSlotName::FILTER, kernel_initializer},
    };
  }
}

} // namespace FlexFlow
