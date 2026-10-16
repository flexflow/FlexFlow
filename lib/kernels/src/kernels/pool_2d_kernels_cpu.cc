#include "kernels/pool_2d_kernels_cpu.h"
#include "kernels/datatype_dispatch.h"
#include "op-attrs/ff_ordered/ff_ordered_transform.h"
#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/tensor_dims_coord.h"
#include "utils/containers/any_of.h"
#include "utils/containers/binary_cartesian_product.h"
#include "utils/containers/maximum.h"
#include "utils/containers/multiset_of.h"
#include "utils/containers/set_of.h"
#include "utils/containers/sum.h"
#include "utils/containers/transform_ref.h"
#include "utils/containers/vector_of.h"
#include "utils/exception.h"
#include "utils/fmt/multiset.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

static std::set<FFOrdered<int>>
    get_input_coords_for_output_coord(TensorDimsCoord const &output_coord,
                                      positive_int kernel_h,
                                      positive_int kernel_w,
                                      positive_int stride_h,
                                      positive_int stride_w,
                                      nonnegative_int padding_h,
                                      nonnegative_int padding_w) {
  return transform(
      set_of(binary_cartesian_product(set_of(nonnegative_range(kernel_h)),
                                      set_of(nonnegative_range(kernel_w)))),
      [&](std::pair<nonnegative_int, nonnegative_int> const &p)
          -> FFOrdered<int> {
        nonnegative_int h_offset = p.first;
        nonnegative_int w_offset = p.second;

        relative_ff_dim_t h_dim = relative_ff_dim_t{-2};
        relative_ff_dim_t w_dim = relative_ff_dim_t{-1};

        auto compute_input_coord = [](nonnegative_int output_component,
                                      positive_int stride,
                                      nonnegative_int offset,
                                      nonnegative_int padding) -> int {
          int out = output_component.int_from_nonnegative_int();
          int s = stride.int_from_positive_int();
          int o = offset.int_from_nonnegative_int();
          int p = padding.int_from_nonnegative_int();

          return out * s + o - p;
        };

        FFOrdered<int> result = ff_ordered_transform(
            output_coord.ff_ordered, [](nonnegative_int x) -> int {
              return x.int_from_nonnegative_int();
            });

        transform_ref(result.at(h_dim), [&](int output_component) -> int {
          return compute_input_coord(
              nonnegative_int{output_component}, stride_h, h_offset, padding_h);
        });

        transform_ref(result.at(w_dim), [&](int output_component) -> int {
          return compute_input_coord(
              nonnegative_int{output_component}, stride_w, w_offset, padding_w);
        });

        return result;
      });
}

static std::optional<TensorDimsCoord>
    contains_potentially_negative_coord(TensorDims const &dims,
                                        FFOrdered<int> const &coord) {
  int contains_negative_coord = any_of(
      vector_of(coord), [&](int component) -> bool { return component < 0; });
  if (contains_negative_coord) {
    return std::nullopt;
  }

  TensorDimsCoord input_coord = TensorDimsCoord{
      ff_ordered_transform(coord,
                           [&](int component) -> nonnegative_int {
                             return nonnegative_int{component};
                           }),
  };

  if (tensor_dims_contains_coord(dims, input_coord)) {
    return input_coord;
  } else {
    return std::nullopt;
  }
}

template <DataType DT>
struct CPUPool2DTensorAccessor {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output,
                  Pool2DAttrs const &attrs) {
    using T = real_type_t<DT>;

    ASSERT(input.device_type == DeviceType::CPU);
    ASSERT(output.device_type == DeviceType::CPU);

    std::function<T(std::multiset<T> const &)> compute_output;
    switch (attrs.pool_type) {
      case PoolOp::MAX:
        compute_output = [&](std::multiset<T> const &inputs) -> T {
          return maximum(inputs);
        };
        break;
      case PoolOp::AVG:
        compute_output = [&](std::multiset<T> const &inputs) -> T {
          return sum(inputs) / inputs.size();
        };
        break;
      default:
        PANIC();
    };

    ASSERT(!attrs.activation.has_value());

    for (TensorDimsCoord const &output_coord :
         get_tensor_dims_coord_set(output.shape.dims)) {

      std::set<FFOrdered<int>> input_coords =
          get_input_coords_for_output_coord(output_coord,
                                            /*kernel_h=*/attrs.kernel_h,
                                            /*kernel_w=*/attrs.kernel_w,
                                            /*stride_h=*/attrs.stride_h,
                                            /*stride_w=*/attrs.stride_w,
                                            /*padding_h=*/attrs.padding_h,
                                            /*padding_w=*/attrs.padding_w);

      std::multiset<T> input_values =
          transform(multiset_of(input_coords),
                    [&](FFOrdered<int> const &input_coord) -> T {
                      std::optional<TensorDimsCoord> in_domain_coord =
                          contains_potentially_negative_coord(input.shape.dims,
                                                              input_coord);

                      return transform(in_domain_coord,
                                       [&](TensorDimsCoord const &c) -> T {
                                         return input.at<DT>(c);
                                       })
                          .value_or(0);
                    });

      output.at<DT>(output_coord) = compute_output(input_values);
    }
  }
};

void pool2d_cpu_forward_kernel(Pool2DAttrs const &attrs,
                               GenericTensorAccessorR const &input,
                               GenericTensorAccessorW const &output) {
  TensorShape correct_output_shape =
      pool2d_get_output_shape(attrs, input.shape);
  ASSERT(output.shape == correct_output_shape);

  DataTypeDispatch1<CPUPool2DTensorAccessor>{}(
      input.shape.data_type, input, output, attrs);
}

void pool2d_cpu_backward_kernel(Pool2DAttrs const &attrs,
                                GenericTensorAccessorR const &output_grad,
                                GenericTensorAccessorW const &input_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
