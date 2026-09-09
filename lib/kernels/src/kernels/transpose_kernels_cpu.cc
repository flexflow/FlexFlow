#include "kernels/transpose_kernels_cpu.h"
#include "kernels/datatype_dispatch.h"
#include "op-attrs/ff_ordered/ff_ordered_from_map.h"
#include "op-attrs/ff_ordered/map_from_ff_ordered.h"
#include "op-attrs/tensor_dim_permutation.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/require_same.h"

namespace FlexFlow {

// The coordinate in the transposed tensor that `input_coord` corresponds to,
// i.e. `output_coord[d] = input_coord[permutation.at_l(d)]`. This is the same
// relabelling `permute_tensor_dims` applies to the dims themselves.
static TensorDimsCoord
    permute_tensor_dims_coord(TensorDimPermutation const &permutation,
                              TensorDimsCoord const &input_coord) {
  return TensorDimsCoord{
      ff_ordered_from_map(
          map_keys(map_from_ff_ordered(input_coord.ff_ordered),
                   [&](ff_dim_t d) { return permutation.at_r(d); })),
  };
}

static void check_shapes(TransposeAttrs const &attrs,
                         TensorShape const &input_shape,
                         TensorShape const &output_shape) {
  require_same(input_shape.data_type, output_shape.data_type);
  ASSERT(permute_tensor_dims(attrs.permutation, input_shape.dims) ==
             output_shape.dims,
         attrs,
         input_shape,
         output_shape);
}

template <DataType DT>
struct TransposeCPUForwardKernel {
  void operator()(TransposeAttrs const &attrs,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) const {
    for (TensorDimsCoord const &input_coord :
         get_tensor_dims_coord_set(input.shape.dims)) {
      output.at<DT>(permute_tensor_dims_coord(attrs.permutation, input_coord)) =
          input.at<DT>(input_coord);
    }
  }
};

void transpose_cpu_forward_kernel(TransposeAttrs const &attrs,
                                  GenericTensorAccessorR const &input,
                                  GenericTensorAccessorW const &output) {
  check_shapes(attrs, input.shape, output.shape);

  DataTypeDispatch1<TransposeCPUForwardKernel>{}(
      input.shape.data_type, attrs, input, output);
}

template <DataType DT>
struct TransposeCPUBackwardKernel {
  void operator()(TransposeAttrs const &attrs,
                  GenericTensorAccessorR const &output_grad,
                  GenericTensorAccessorW const &input_grad) const {
    for (TensorDimsCoord const &input_coord :
         get_tensor_dims_coord_set(input_grad.shape.dims)) {
      input_grad.at<DT>(input_coord) += output_grad.at<DT>(
          permute_tensor_dims_coord(attrs.permutation, input_coord));
    }
  }
};

void transpose_cpu_backward_kernel(TransposeAttrs const &attrs,
                                   GenericTensorAccessorR const &output,
                                   GenericTensorAccessorR const &output_grad,
                                   GenericTensorAccessorR const &input,
                                   GenericTensorAccessorW const &input_grad) {
  check_shapes(attrs, input_grad.shape, output_grad.shape);

  DataTypeDispatch1<TransposeCPUBackwardKernel>{}(
      input_grad.shape.data_type, attrs, output_grad, input_grad);
}

} // namespace FlexFlow
