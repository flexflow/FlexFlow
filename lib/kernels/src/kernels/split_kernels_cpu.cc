#include "kernels/split_kernels_cpu.h"
#include "kernels/datatype_dispatch.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_dims_coord.h"
#include "utils/containers/require_same.h"

namespace FlexFlow {

// The coordinate in the tensor being split that `output_coord` (a coordinate
// in the output starting at `offset` along `axis`) corresponds to.
static TensorDimsCoord
    input_coord_from_output_coord(TensorDimsCoord const &output_coord,
                                  ff_dim_t axis,
                                  nonnegative_int offset) {
  TensorDimsCoord input_coord = output_coord;
  tensor_dims_coord_at_idx(input_coord, axis) += offset;
  return input_coord;
}

// Checks that the outputs agree with the input on every dim but `axis`, and
// that their sizes along `axis` are the requested splits and add up to the
// input's.
static void check_shapes(SplitAttrs const &attrs,
                         TensorShape const &input_shape,
                         std::vector<TensorShape> const &output_shapes) {
  ASSERT(output_shapes.size() == attrs.splits.size());

  nonnegative_int total = 0_n;
  for (int i = 0; i < output_shapes.size(); i++) {
    TensorShape const &output_shape = output_shapes.at(i);

    require_same(input_shape.data_type, output_shape.data_type);
    ASSERT(
        tensor_dims_drop_dims(output_shape.dims,
                              [&](ff_dim_t d) { return d == attrs.axis; }) ==
            tensor_dims_drop_dims(input_shape.dims,
                                  [&](ff_dim_t d) { return d == attrs.axis; }),
        input_shape,
        output_shape,
        attrs.axis);
    ASSERT(dim_at_idx(output_shape.dims, attrs.axis) == attrs.splits.at(i));

    total += dim_at_idx(output_shape.dims, attrs.axis)
                 .nonnegative_int_from_positive_int();
  }

  ASSERT(total == dim_at_idx(input_shape.dims, attrs.axis)
                      .nonnegative_int_from_positive_int());
}

template <DataType DT>
struct SplitCPUForwardKernel {
  void operator()(SplitAttrs const &attrs,
                  GenericTensorAccessorR const &input,
                  std::vector<GenericTensorAccessorW> const &outputs) const {
    nonnegative_int offset = 0_n;

    for (GenericTensorAccessorW const &output : outputs) {
      for (TensorDimsCoord const &output_coord :
           get_tensor_dims_coord_set(output.shape.dims)) {
        output.at<DT>(output_coord) = input.at<DT>(
            input_coord_from_output_coord(output_coord, attrs.axis, offset));
      }

      offset += dim_at_idx(output.shape.dims, attrs.axis)
                    .nonnegative_int_from_positive_int();
    }
  }
};

void split_cpu_forward_kernel(
    SplitAttrs const &attrs,
    GenericTensorAccessorR const &input,
    std::vector<GenericTensorAccessorW> const &outputs) {
  check_shapes(attrs,
               input.shape,
               transform(outputs, [](GenericTensorAccessorW const &output) {
                 return output.shape;
               }));

  DataTypeDispatch1<SplitCPUForwardKernel>{}(
      input.shape.data_type, attrs, input, outputs);
}

template <DataType DT>
struct SplitCPUBackwardKernel {
  void operator()(SplitAttrs const &attrs,
                  std::vector<GenericTensorAccessorR> const &output_grads,
                  GenericTensorAccessorW const &input_grad) const {
    nonnegative_int offset = 0_n;

    for (GenericTensorAccessorR const &output_grad : output_grads) {
      for (TensorDimsCoord const &output_coord :
           get_tensor_dims_coord_set(output_grad.shape.dims)) {
        input_grad.at<DT>(
            input_coord_from_output_coord(output_coord, attrs.axis, offset)) +=
            output_grad.at<DT>(output_coord);
      }

      offset += dim_at_idx(output_grad.shape.dims, attrs.axis)
                    .nonnegative_int_from_positive_int();
    }
  }
};

void split_cpu_backward_kernel(
    SplitAttrs const &attrs,
    std::vector<GenericTensorAccessorR> const &outputs,
    std::vector<GenericTensorAccessorR> const &output_grads,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad) {
  check_shapes(
      attrs,
      input_grad.shape,
      transform(output_grads, [](GenericTensorAccessorR const &output_grad) {
        return output_grad.shape;
      }));

  DataTypeDispatch1<SplitCPUBackwardKernel>{}(
      input_grad.shape.data_type, attrs, output_grads, input_grad);
}

} // namespace FlexFlow
