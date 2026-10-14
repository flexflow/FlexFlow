#include "kernels/conv_2d_kernels_cpu.h"
#include "kernels/datatype_dispatch.h"
#include "op-attrs/ff_ordered/ff_ordered_from_map.h"
#include "op-attrs/ff_ordered/map_from_ff_ordered.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/tensor_dims_coord.h"
#include "utils/containers/get_all_assignments.h"
#include "utils/containers/product.h"
#include "utils/containers/set_of.h"
#include "utils/containers/sorted.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip_values_strict_with.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

// TODO(@lockshaw): should be folded into utils/orthotope
static std::vector<TensorDimsCoord>
    points_in_range(TensorDimsCoord const &min_coord,
                    TensorDimsCoord const &max_coord) {
  std::map<ff_dim_t, nonnegative_int> min_coord_map =
      map_from_ff_ordered(min_coord.ff_ordered);
  std::map<ff_dim_t, nonnegative_int> max_coord_map =
      map_from_ff_ordered(max_coord.ff_ordered);

  std::map<ff_dim_t, positive_int> deltas = zip_values_strict_with(
      min_coord_map,
      max_coord_map,
      [&](nonnegative_int min, nonnegative_int max) -> positive_int {
        ASSERT(min < max);
        return positive_int{
            max.int_from_nonnegative_int() - min.int_from_nonnegative_int(),
        };
      });

  std::map<ff_dim_t, std::set<nonnegative_int>> dim_ranges =
      zip_values_strict_with(min_coord_map,
                             max_coord_map,
                             [&](nonnegative_int min, nonnegative_int max)
                                 -> std::set<nonnegative_int> {
                               ASSERT(min < max);
                               return set_of(nonnegative_range(min, max));
                             });

  std::vector<TensorDimsCoord> result = sorted(transform(
      get_all_assignments(dim_ranges),
      [&](std::map<ff_dim_t, nonnegative_int> const &p) -> TensorDimsCoord {
        return TensorDimsCoord{
            ff_ordered_from_map(p),
        };
      }));

  {
    int expected_num_points = product(values(deltas)).int_from_positive_int();
    int actual_num_points = result.size();
    ASSERT(actual_num_points == expected_num_points);
  }

  return result;
}

static std::pair<TensorDimsCoord, TensorDimsCoord>
    get_input_coord_interval_for_output_coord(
        TensorDimsCoord const &output_coord,
        positive_int num_input_channels,
        positive_int kernel_h,
        positive_int kernel_w,
        positive_int stride_h,
        positive_int stride_w) {
  nonnegative_int output_batch_component =
      tensor_dims_coord_at_idx(output_coord, ff_dim_t{0_n});
  nonnegative_int output_channel_component =
      tensor_dims_coord_at_idx(output_coord, ff_dim_t{1_n});
  nonnegative_int output_height_component =
      tensor_dims_coord_at_idx(output_coord, ff_dim_t{2_n});
  nonnegative_int output_width_component =
      tensor_dims_coord_at_idx(output_coord, ff_dim_t{3_n});

  relative_ff_dim_t h_dim = relative_ff_dim_t{-2};
  relative_ff_dim_t w_dim = relative_ff_dim_t{-1};

  TensorDimsCoord min_coord = TensorDimsCoord{
      FFOrdered<nonnegative_int>{
          output_batch_component,
          0_n,
          output_height_component * stride_h,
          output_width_component * stride_w,
      },
  };

  TensorDimsCoord max_coord = TensorDimsCoord{
      FFOrdered<nonnegative_int>{
          output_batch_component + 1_n,
          num_input_channels.nonnegative_int_from_positive_int(),
          (output_height_component * stride_h + kernel_h)
              .nonnegative_int_from_positive_int(),
          (output_width_component * stride_w + kernel_w)
              .nonnegative_int_from_positive_int(),
      },
  };

  return std::pair{
      min_coord,
      max_coord,
  };
}

static TensorDimsCoord
    get_bias_coord_for_output_coord(TensorDimsCoord const &output_coord) {
  ff_dim_t output_channel_dim = ff_dim_t{1_n};

  nonnegative_int output_channel_component =
      tensor_dims_coord_at_idx(output_coord, output_channel_dim);

  return TensorDimsCoord{
      FFOrdered<nonnegative_int>{
          output_channel_component,
      },
  };
}

static std::pair<TensorDimsCoord, TensorDimsCoord>
    get_kernel_coord_interval_for_output_coord(
        TensorDimsCoord const &output_coord,
        positive_int num_input_channels,
        positive_int kernel_h,
        positive_int kernel_w,
        positive_int stride_h,
        positive_int stride_w) {
  ff_dim_t output_channel_dim = ff_dim_t{1_n};

  nonnegative_int output_channel_component =
      tensor_dims_coord_at_idx(output_coord, output_channel_dim);

  TensorDimsCoord min_coord = TensorDimsCoord{
      FFOrdered<nonnegative_int>{
          output_channel_component,
          0_n,
          0_n,
          0_n,
      },
  };

  TensorDimsCoord max_coord = TensorDimsCoord{
      FFOrdered<nonnegative_int>{
          output_channel_component + 1_n,
          num_input_channels.nonnegative_int_from_positive_int(),
          kernel_h.nonnegative_int_from_positive_int(),
          kernel_w.nonnegative_int_from_positive_int(),
      },
  };

  return std::pair{
      min_coord,
      max_coord,
  };
}

template <DataType DT>
struct CPUConv2DTensorAccessor {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorR const &filter,
                  std::optional<GenericTensorAccessorR> const &bias,
                  GenericTensorAccessorW const &output,
                  Conv2DAttrs const &attrs) {
    using T = real_type_t<DT>;

    ASSERT(!attrs.activation.has_value());
    ASSERT(attrs.use_bias == bias.has_value());

    ASSERT(input.device_type == DeviceType::CPU);
    ASSERT(filter.device_type == DeviceType::CPU);
    if (bias.has_value()) {
      ASSERT(bias.value().device_type == DeviceType::CPU);
    }
    ASSERT(output.device_type == DeviceType::CPU);

    positive_int num_input_channels =
        dim_at_idx(input.shape.dims, ff_dim_t{1_n});
    positive_int num_output_channels =
        dim_at_idx(output.shape.dims, ff_dim_t{1_n});

    for (TensorDimsCoord const &output_coord :
         get_tensor_dims_coord_set(output.shape.dims)) {

      std::pair<TensorDimsCoord, TensorDimsCoord> input_coord_interval =
          get_input_coord_interval_for_output_coord(
              /*output_coord=*/output_coord,
              /*num_input_channels=*/num_input_channels,
              /*kernel_h=*/attrs.kernel_h,
              /*kernel_w=*/attrs.kernel_w,
              /*stride_h=*/attrs.stride_h,
              /*stride_w=*/attrs.stride_w);

      std::pair<TensorDimsCoord, TensorDimsCoord> kernel_coord_interval =
          get_kernel_coord_interval_for_output_coord(
              /*output_coord=*/output_coord,
              /*num_input_channels=*/num_input_channels,
              /*kernel_h=*/attrs.kernel_h,
              /*kernel_w=*/attrs.kernel_w,
              /*stride_h=*/attrs.stride_h,
              /*stride_w=*/attrs.stride_w);

      T result =
          sum(zip_with_strict(points_in_range(input_coord_interval.first,
                                              input_coord_interval.second),
                              points_in_range(kernel_coord_interval.first,
                                              kernel_coord_interval.second),
                              [&](TensorDimsCoord const &input_coord,
                                  TensorDimsCoord const &filter_coord) -> T {
                                return input.at<DT>(input_coord) *
                                       filter.at<DT>(filter_coord);
                              }));

      if (bias.has_value()) {
        TensorDimsCoord bias_coord =
            get_bias_coord_for_output_coord(output_coord);
        result += bias.value().at<DT>(bias_coord);
      }

      output.at<DT>(output_coord) = result;
    }
  }
};

void conv2d_cpu_forward_kernel(
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &filter,
    std::optional<GenericTensorAccessorR> const &bias,
    GenericTensorAccessorW const &output) {
  {
    TensorShape correct_output_shape =
        conv2d_get_output_shape(attrs, input.shape);
    ASSERT(output.shape == correct_output_shape);
  }

  {
    TensorShape correct_filter_shape =
        conv2d_get_kernel_shape(attrs, input.shape);
    ASSERT(filter.shape == correct_filter_shape);
  }

  {
    std::optional<TensorShape> correct_bias_shape =
        conv2d_get_bias_shape(attrs, input.shape);
    std::optional<TensorShape> bias_shape =
        transform(bias, [](GenericTensorAccessorR const &b) -> TensorShape {
          return b.shape;
        });
    ASSERT(bias_shape == correct_bias_shape);
  }

  DataTypeDispatch1<CPUConv2DTensorAccessor>{}(
      input.shape.data_type, input, filter, bias, output, attrs);
}

void conv2d_cpu_backward_kernel(
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &filter,
    GenericTensorAccessorW const &filter_grad,
    std::optional<GenericTensorAccessorR> const &bias,
    std::optional<GenericTensorAccessorW> const &bias_grad,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
