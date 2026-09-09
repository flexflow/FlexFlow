#include "kernels/upsample_kernels_cpu.h"
#include "kernels/datatype_dispatch.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_dims_coord.h"
#include "utils/containers/require_same.h"

namespace FlexFlow {

template <DataType DT>
struct UpsampleCPUForwardKernel {
  void operator()(UpsampleAttrs const &attrs,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    auto input_coord_from_output_coord =
        [&](TensorDimsCoord const &output_coord) -> TensorDimsCoord {
      TensorDimsCoord input_coord = output_coord;
      tensor_dims_coord_at_rel_idx(input_coord, relative_ff_dim_t{-1}) /=
          attrs.scale_factor;
      tensor_dims_coord_at_rel_idx(input_coord, relative_ff_dim_t{-2}) /=
          attrs.scale_factor;
      return input_coord;
    };

    for (TensorDimsCoord output_coord :
         get_tensor_dims_coord_set(output.shape.dims)) {
      TensorDimsCoord input_coord = input_coord_from_output_coord(output_coord);

      output.at<DT>(output_coord) = input.at<DT>(input_coord);
    }
  }
};

void upsample_cpu_forward_kernel(UpsampleAttrs const &attrs,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &output) {
  ASSERT(get_num_dims(input.shape.dims) == num_tensor_dims_t{4_n},
         "Currently Upsample only supports 4-dimensional input tensors (i.e., "
         "NCHW tensors). "
         "If you need other support for other tensor shapes, please create an "
         "issue.");

  ASSERT(attrs.mode == UpsampleMode::NEAREST,
         "Currently Upsample is only supports mode {}. "
         "If you need other support for other modes, please create an issue.",
         UpsampleMode::NEAREST);

  DataType data_type =
      require_same(input.shape.data_type, output.shape.data_type);

  DataTypeDispatch1<UpsampleCPUForwardKernel>{}(
      data_type, attrs, input, output);
}

template <DataType DT>
struct UpsampleCPUBackwardKernel {
  void operator()(UpsampleAttrs const &attrs,
                  GenericTensorAccessorR const &output_grad,
                  GenericTensorAccessorW const &input_grad) {
    auto input_coord_from_output_coord =
        [&](TensorDimsCoord const &output_coord) -> TensorDimsCoord {
      TensorDimsCoord input_coord = output_coord;
      tensor_dims_coord_at_rel_idx(input_coord, relative_ff_dim_t{-1}) /=
          attrs.scale_factor;
      tensor_dims_coord_at_rel_idx(input_coord, relative_ff_dim_t{-2}) /=
          attrs.scale_factor;
      return input_coord;
    };

    for (TensorDimsCoord output_coord :
         get_tensor_dims_coord_set(output_grad.shape.dims)) {
      TensorDimsCoord input_coord = input_coord_from_output_coord(output_coord);

      input_grad.at<DT>(input_coord) += output_grad.at<DT>(output_coord);
    }
  }
};

void upsample_cpu_backward_kernel(UpsampleAttrs const &attrs,
                                  GenericTensorAccessorR const &output,
                                  GenericTensorAccessorR const &output_grad,
                                  GenericTensorAccessorR const &input,
                                  GenericTensorAccessorW const &input_grad) {
  ASSERT(get_num_dims(input_grad.shape.dims) == num_tensor_dims_t{4_n},
         "Currently Upsample only supports 4-dimensional input tensors (i.e., "
         "NCHW tensors). "
         "If you need other support for other tensor shapes, please create an "
         "issue.");

  ASSERT(attrs.mode == UpsampleMode::NEAREST,
         "Currently Upsample is only supports mode {}. "
         "If you need other support for other modes, please create an issue.",
         UpsampleMode::NEAREST);

  DataType data_type =
      require_same(output_grad.shape.data_type, input_grad.shape.data_type);

  DataTypeDispatch1<UpsampleCPUBackwardKernel>{}(
      data_type, attrs, output_grad, input_grad);
}

} // namespace FlexFlow
