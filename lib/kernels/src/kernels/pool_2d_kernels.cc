#include "kernels/pool_2d_kernels.h"
#include "kernels/pool_2d_kernels_cpu.h"
#include "kernels/pool_2d_kernels_gpu.h"
#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/tensor_dims.h"
#include "utils/exception.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

std::optional<Pool2DPerDeviceState>
    pool2d_init_kernel(DeviceType device_type,
                       device_handle_t const &handle,
                       Pool2DAttrs const &attrs,
                       TensorShape const &input_shape) {
  if (device_type == DeviceType::GPU) {
    TensorDims input_dims = input_shape.dims;
    TensorDims output_dims = pool2d_get_output_shape(attrs, input_shape).dims;

    ff_dim_t n_dim = ff_dim_t{0_n};
    ff_dim_t c_dim = ff_dim_t{1_n};
    ff_dim_t h_dim = ff_dim_t{2_n};
    ff_dim_t w_dim = ff_dim_t{3_n};

    auto get_dim = [](TensorDims const &dims, ff_dim_t dim_idx) -> int {
      return dim_at_idx(dims, dim_idx).int_from_positive_int();
    };

    return pool2d_gpu_init_kernel(
        /*handle=*/handle.require_for_gpu(),
        /*activation=*/attrs.activation,
        /*input_w=*/get_dim(input_dims, w_dim),
        /*input_h=*/get_dim(input_dims, h_dim),
        /*input_c=*/get_dim(input_dims, c_dim),
        /*input_n=*/get_dim(input_dims, n_dim),
        /*output_w=*/get_dim(output_dims, w_dim),
        /*output_h=*/get_dim(output_dims, h_dim),
        /*output_c=*/get_dim(output_dims, c_dim),
        /*output_n=*/get_dim(output_dims, n_dim),
        /*pad_h=*/attrs.padding_h.int_from_nonnegative_int(),
        /*pad_w=*/attrs.padding_w.int_from_nonnegative_int(),
        /*kernel_h=*/attrs.kernel_h.int_from_positive_int(),
        /*kernel_w=*/attrs.kernel_w.int_from_positive_int(),
        /*stride_h=*/attrs.stride_h.int_from_positive_int(),
        /*stride_w=*/attrs.stride_w.int_from_positive_int(),
        /*pool_type=*/attrs.pool_type);
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(handle.is_for_cpu());
    return std::nullopt;
  }
}

void pool2d_forward_kernel(
    device_stream_t const &stream,
    std::optional<Pool2DPerDeviceState> const &per_device_state,
    Pool2DAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output) {
  if (stream.is_gpu()) {
    pool2d_gpu_forward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*input_ptr=*/input.get_float_ptr(),
        /*output_ptr=*/output.get_float_ptr());
  } else {
    ASSERT(stream.is_cpu());
    pool2d_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/output);
  }
}

void pool2d_backward_kernel(
    device_stream_t const &stream,
    std::optional<Pool2DPerDeviceState> const &per_device_state,
    Pool2DAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad) {
  if (stream.is_gpu()) {
    pool2d_gpu_backward_kernel(
        /*stream=*/stream.require_gpu(),
        /*per_device_state=*/per_device_state.value(),
        /*output_ptr=*/output.ptr,
        /*output_grad_ptr=*/output_grad.ptr,
        /*input_ptr=*/input.ptr,
        /*input_grad_ptr=*/input_grad.ptr);
  } else {
    ASSERT(stream.is_cpu());
    pool2d_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*output_grad=*/output_grad,
        /*input_grad=*/input_grad);
  }
}

void pool2d_cleanup_kernel(
    DeviceType device_type,
    std::optional<Pool2DPerDeviceState> &per_device_state) {
  if (device_type == DeviceType::GPU) {
    pool2d_gpu_cleanup_kernel(per_device_state.value());
  } else {
    ASSERT(device_type == DeviceType::CPU);
    ASSERT(per_device_state == std::nullopt);
  }
}

} // namespace FlexFlow
