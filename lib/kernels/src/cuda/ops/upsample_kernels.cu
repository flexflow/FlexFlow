#include "internal/device.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/upsample_kernels_gpu.h"
#include "op-attrs/tensor_dims.h"
#include "utils/containers/require_same.h"

namespace FlexFlow {

// The shape of an NCHW tensor, flattened into the quantities the kernels below
// need in order to turn a linear index into coordinates.
struct UpsampleNCHWDims {
  int num_samples;
  int num_channels;
  int height;
  int width;
};

static UpsampleNCHWDims get_nchw_dims(TensorShape const &shape) {
  return UpsampleNCHWDims{
      /*num_samples=*/
      dim_at_idx(shape.dims, ff_dim_t{0_n}).int_from_positive_int(),
      /*num_channels=*/
      dim_at_idx(shape.dims, ff_dim_t{1_n}).int_from_positive_int(),
      /*height=*/dim_at_idx(shape.dims, ff_dim_t{2_n}).int_from_positive_int(),
      /*width=*/dim_at_idx(shape.dims, ff_dim_t{3_n}).int_from_positive_int(),
  };
}

static void check_shapes(UpsampleAttrs const &attrs,
                         TensorShape const &input_shape,
                         TensorShape const &output_shape) {
  ASSERT(get_num_dims(input_shape.dims) == num_tensor_dims_t{4_n},
         "Currently Upsample only supports 4-dimensional input tensors (i.e., "
         "NCHW tensors). "
         "If you need other support for other tensor shapes, please create an "
         "issue.",
         input_shape);

  ASSERT(attrs.mode == UpsampleMode::NEAREST,
         "Currently Upsample is only supports mode {}. "
         "If you need other support for other modes, please create an issue.",
         UpsampleMode::NEAREST);

  UpsampleNCHWDims input_dims = get_nchw_dims(input_shape);
  UpsampleNCHWDims output_dims = get_nchw_dims(output_shape);
  int scale_factor = attrs.scale_factor.int_from_int_ge_two();

  ASSERT(output_dims.num_samples == input_dims.num_samples);
  ASSERT(output_dims.num_channels == input_dims.num_channels);
  ASSERT(output_dims.height == input_dims.height * scale_factor);
  ASSERT(output_dims.width == input_dims.width * scale_factor);
}

// Each thread computes one output element by reading the input element the
// output coordinate maps back to.
template <typename T>
__global__ void upsample_nearest_forward_kernel(size_t num_output_elements,
                                                T const *input,
                                                T *output,
                                                UpsampleNCHWDims input_dims,
                                                int scale_factor) {
  CUDA_KERNEL_LOOP(output_idx, num_output_elements) {
    int output_width = input_dims.width * scale_factor;
    int output_height = input_dims.height * scale_factor;

    int w = output_idx % output_width;
    int h = (output_idx / output_width) % output_height;
    int nc = output_idx / (output_width * output_height);

    int input_idx =
        (nc * input_dims.height + h / scale_factor) * input_dims.width +
        w / scale_factor;

    output[output_idx] = input[input_idx];
  }
}

// Each thread accumulates one input element by summing over the block of
// output elements that read from it. Iterating over inputs (rather than
// outputs) keeps the accumulation deterministic and avoids atomics.
template <typename T>
__global__ void upsample_nearest_backward_kernel(size_t num_input_elements,
                                                 T const *output_grad,
                                                 T *input_grad,
                                                 UpsampleNCHWDims input_dims,
                                                 int scale_factor) {
  CUDA_KERNEL_LOOP(input_idx, num_input_elements) {
    int output_width = input_dims.width * scale_factor;
    int output_height = input_dims.height * scale_factor;

    int w = input_idx % input_dims.width;
    int h = (input_idx / input_dims.width) % input_dims.height;
    int nc = input_idx / (input_dims.width * input_dims.height);

    T sum = 0;
    for (int dh = 0; dh < scale_factor; dh++) {
      for (int dw = 0; dw < scale_factor; dw++) {
        int output_idx =
            (nc * output_height + h * scale_factor + dh) * output_width +
            w * scale_factor + dw;
        sum += output_grad[output_idx];
      }
    }

    input_grad[input_idx] += sum;
  }
}

template <DataType DT>
struct UpsampleGPUForwardKernel {
  void operator()(cudaStream_t stream,
                  UpsampleAttrs const &attrs,
                  GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) const {
    int num_output_elements =
        get_num_elements(output.shape.dims).int_from_positive_int();

    upsample_nearest_forward_kernel<real_type_t<DT>>
        <<<GET_BLOCKS(num_output_elements), CUDA_NUM_THREADS, 0, stream>>>(
            num_output_elements,
            input.get<DT>(),
            output.get<DT>(),
            get_nchw_dims(input.shape),
            attrs.scale_factor.int_from_int_ge_two());
  }
};

void upsample_gpu_forward_kernel(cudaStream_t stream,
                                 UpsampleAttrs const &attrs,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &output) {
  check_shapes(attrs, input.shape, output.shape);

  DataType data_type =
      require_same(input.shape.data_type, output.shape.data_type);

  DataTypeDispatch1<UpsampleGPUForwardKernel>{}(
      data_type, stream, attrs, input, output);
}

template <DataType DT>
struct UpsampleGPUBackwardKernel {
  void operator()(cudaStream_t stream,
                  UpsampleAttrs const &attrs,
                  GenericTensorAccessorR const &output_grad,
                  GenericTensorAccessorW const &input_grad) const {
    int num_input_elements =
        get_num_elements(input_grad.shape.dims).int_from_positive_int();

    upsample_nearest_backward_kernel<real_type_t<DT>>
        <<<GET_BLOCKS(num_input_elements), CUDA_NUM_THREADS, 0, stream>>>(
            num_input_elements,
            output_grad.get<DT>(),
            input_grad.get<DT>(),
            get_nchw_dims(input_grad.shape),
            attrs.scale_factor.int_from_int_ge_two());
  }
};

void upsample_gpu_backward_kernel(cudaStream_t stream,
                                  UpsampleAttrs const &attrs,
                                  GenericTensorAccessorR const &output,
                                  GenericTensorAccessorR const &output_grad,
                                  GenericTensorAccessorR const &input,
                                  GenericTensorAccessorW const &input_grad) {
  check_shapes(attrs, input_grad.shape, output_grad.shape);

  DataType data_type =
      require_same(output_grad.shape.data_type, input_grad.shape.data_type);

  DataTypeDispatch1<UpsampleGPUBackwardKernel>{}(
      data_type, stream, attrs, output_grad, input_grad);
}

} // namespace FlexFlow
