#include "kernels/concat_kernels_cpu.h"
#include "kernels/datatype_dispatch.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_dims_coord.h"
#include "utils/containers/require_same.h"

namespace FlexFlow {

// The coordinate in the concatenated tensor that `input_coord` (a coordinate
// in the input starting at `offset` along `axis`) corresponds to.
static TensorDimsCoord output_coord_from_input_coord(
    TensorDimsCoord const &input_coord, ff_dim_t axis, nonnegative_int offset) {
  TensorDimsCoord output_coord = input_coord;
  tensor_dims_coord_at_idx(output_coord, axis) += offset;
  return output_coord;
}

// Checks that the inputs agree with the output on every dim but `axis`, and
// that their sizes along `axis` add up to the output's.
static void check_shapes(ConcatAttrs const &attrs,
                         std::vector<TensorShape> const &input_shapes,
                         TensorShape const &output_shape) {
  ASSERT(input_shapes.size() == attrs.num_inputs.int_from_int_ge_two());

  nonnegative_int total = 0_n;
  for (TensorShape const &input_shape : input_shapes) {
    require_same(input_shape.data_type, output_shape.data_type);
    ASSERT(
        tensor_dims_drop_dims(input_shape.dims,
                              [&](ff_dim_t d) { return d == attrs.axis; }) ==
            tensor_dims_drop_dims(output_shape.dims,
                                  [&](ff_dim_t d) { return d == attrs.axis; }),
        input_shape,
        output_shape,
        attrs.axis);
    total += dim_at_idx(input_shape.dims, attrs.axis)
                 .nonnegative_int_from_positive_int();
  }

  ASSERT(total == dim_at_idx(output_shape.dims, attrs.axis)
                      .nonnegative_int_from_positive_int());
}

template <DataType DT>
struct ConcatCPUForwardKernel {
  void operator()(ConcatAttrs const &attrs,
                  std::vector<GenericTensorAccessorR> const &inputs,
                  GenericTensorAccessorW const &output) const {
    nonnegative_int offset = 0_n;

    for (GenericTensorAccessorR const &input : inputs) {
      for (TensorDimsCoord const &input_coord :
           get_tensor_dims_coord_set(input.shape.dims)) {
        output.at<DT>(output_coord_from_input_coord(
            input_coord, attrs.axis, offset)) = input.at<DT>(input_coord);
      }

      offset += dim_at_idx(input.shape.dims, attrs.axis)
                    .nonnegative_int_from_positive_int();
    }
  }
};

void concat_cpu_forward_kernel(
    ConcatAttrs const &attrs,
    std::vector<GenericTensorAccessorR> const &inputs,
    GenericTensorAccessorW const &output) {
  check_shapes(attrs,
               transform(inputs,
                         [](GenericTensorAccessorR const &input) {
                           return input.shape;
                         }),
               output.shape);

  DataTypeDispatch1<ConcatCPUForwardKernel>{}(
      output.shape.data_type, attrs, inputs, output);
}

template <DataType DT>
struct ConcatCPUBackwardKernel {
  void
      operator()(ConcatAttrs const &attrs,
                 GenericTensorAccessorR const &output_grad,
                 std::vector<GenericTensorAccessorW> const &input_grads) const {
    nonnegative_int offset = 0_n;

    for (GenericTensorAccessorW const &input_grad : input_grads) {
      for (TensorDimsCoord const &input_coord :
           get_tensor_dims_coord_set(input_grad.shape.dims)) {
        input_grad.at<DT>(input_coord) += output_grad.at<DT>(
            output_coord_from_input_coord(input_coord, attrs.axis, offset));
      }

      offset += dim_at_idx(input_grad.shape.dims, attrs.axis)
                    .nonnegative_int_from_positive_int();
    }
  }
};

void concat_cpu_backward_kernel(
    ConcatAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    std::vector<GenericTensorAccessorR> const &inputs,
    std::vector<GenericTensorAccessorW> const &input_grads) {
  check_shapes(attrs,
               transform(input_grads,
                         [](GenericTensorAccessorW const &input_grad) {
                           return input_grad.shape;
                         }),
               output_grad.shape);

  DataTypeDispatch1<ConcatCPUBackwardKernel>{}(
      output_grad.shape.data_type, attrs, output_grad, input_grads);
}

} // namespace FlexFlow
