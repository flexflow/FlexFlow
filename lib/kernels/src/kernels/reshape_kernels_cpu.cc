#include "kernels/reshape_kernels_cpu.h"
#include "kernels/datatype_dispatch.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/require_same.h"

namespace FlexFlow {

// Reshape does not move any data: the input and output have different shapes
// but the same contiguous layout, so viewing one with the other's shape lines
// their elements up.
static GenericTensorAccessorW view_with_shape(GenericTensorAccessorW const &acc,
                                              TensorShape const &shape) {
  return GenericTensorAccessorW{
      /*shape=*/shape,
      /*ptr=*/acc.ptr,
      /*device_type=*/acc.device_type,
  };
}

static void check_shapes(TensorShape const &input_shape,
                         TensorShape const &output_shape) {
  require_same(input_shape.data_type, output_shape.data_type);
  require_same(get_num_elements(input_shape.dims),
               get_num_elements(output_shape.dims));
}

void reshape_cpu_forward_kernel(GenericTensorAccessorR const &input,
                                GenericTensorAccessorW const &output) {
  check_shapes(input.shape, output.shape);

  copy_accessor_data_to_l_from_r(output, input);
}

template <DataType DT>
struct ReshapeCPUBackwardKernel {
  void operator()(GenericTensorAccessorR const &output_grad,
                  GenericTensorAccessorW const &input_grad) const {
    GenericTensorAccessorW input_grad_view =
        view_with_shape(input_grad, output_grad.shape);

    for (TensorDimsCoord const &coord :
         get_tensor_dims_coord_set(output_grad.shape.dims)) {
      input_grad_view.at<DT>(coord) += output_grad.at<DT>(coord);
    }
  }
};

void reshape_cpu_backward_kernel(GenericTensorAccessorR const &output,
                                 GenericTensorAccessorR const &output_grad,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &input_grad) {
  check_shapes(input_grad.shape, output_grad.shape);

  DataTypeDispatch1<ReshapeCPUBackwardKernel>{}(
      output_grad.shape.data_type, output_grad, input_grad);
}

} // namespace FlexFlow
