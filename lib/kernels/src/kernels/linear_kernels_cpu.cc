#include "kernels/linear_kernels_cpu.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/map_tensor_accessors.h"
#include "kernels/tensor_accessor_binary_ops.h"
#include "kernels/tensor_accessor_unary_ops.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

void linear_cpu_forward_kernel(
    LinearAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &output,
    GenericTensorAccessorR const &projection,
    std::optional<GenericTensorAccessorR> const &bias) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  tensor_accessor_matmul_to(
      input, tensor_accessor_transpose(projection, cpu_allocator), output);

  ASSERT(attrs.use_bias == bias.has_value());
  if (bias.has_value()) {
    GenericTensorAccessorW broadcasted_bias = tensor_accessor_broadcast(
        bias.value(), output.shape.dims, cpu_allocator);
    tensor_accessor_elementwise_add_to(
        read_only_accessor_from_write_accessor(output),
        read_only_accessor_from_write_accessor(broadcasted_bias),
        output);
  }

  if (attrs.activation.has_value()) {
    switch (attrs.activation.value()) {
      case Activation::RELU:
        tensor_accessor_relu_to(read_only_accessor_from_write_accessor(output),
                                output);
        break;
      default:
        PANIC("Unhandled activation function", attrs.activation.value());
    }
  }
}

// template <typename T>
static float single_element_relu_bwd(float elem) {
  if (elem > 0) {
    return 1;
  } else {
    return 0;
  }
}

void linear_cpu_backward_kernel(
    LinearAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &projection,
    GenericTensorAccessorW const &projection_grad,
    std::optional<GenericTensorAccessorW> const &bias_grad) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  std::optional<GenericTensorAccessorR> processed_output_grad = std::nullopt;
  if (attrs.activation.has_value()) {
    switch (attrs.activation.value()) {
      case Activation::RELU:
        processed_output_grad =
            read_only_accessor_from_write_accessor(map_tensor_accessor(
                output_grad, single_element_relu_bwd, cpu_allocator));
        break;
      default:
        PANIC("Unhandled activation function", attrs.activation.value());
    }
  } else {
    processed_output_grad = output_grad;
  }

  tensor_accessor_matmul_to(
      processed_output_grad.value(), projection, input_grad);
  tensor_accessor_transpose_to(
      tensor_accessor_matmul(
          read_only_accessor_from_write_accessor(
              tensor_accessor_transpose(input, cpu_allocator)),
          processed_output_grad.value(),
          cpu_allocator),
      projection_grad);

  if (bias_grad.has_value()) {
    tensor_accessor_reduce_to(
        processed_output_grad.value(), ff_dim_t{0_n}, bias_grad.value());
  }
}

} // namespace FlexFlow
