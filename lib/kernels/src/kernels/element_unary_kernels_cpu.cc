#include "kernels/element_unary_kernels_cpu.h"
#include "kernels/map_tensor_accessors.h"
#include "kernels/tensor_accessor_binary_ops.h"
#include <cmath>

namespace FlexFlow {

void element_unary_cpu_forward_kernel(ElementUnaryAttrs const &attrs,
                                      GenericTensorAccessorR const &input,
                                      GenericTensorAccessorW const &output) {
  std::function<float(float)> element_function =
      [&]() -> std::function<float(float)> {
    switch (attrs.op) {
      case ElementUnaryOp::RELU:
        return [](float x) -> float { return std::max(0.0f, x); };
      case ElementUnaryOp::SIGMOID:
        return [](float x) -> float { return (1.0 / (1.0 + expf(-1.0 * x))); };
      case ElementUnaryOp::TANH:
        return [](float x) -> float { return tanhf(x); };
      case ElementUnaryOp::GELU:
        return
            [](float x) -> float { return (x * 0.5 * erfc(-x * M_SQRT1_2)); };
      case ElementUnaryOp::ELU: {
        float alpha = attrs.scalar.value();
        return [alpha](float x) -> float {
          if (x > 0) {
            return x;
          } else {
            return alpha * (expf(x) - 1.0f);
          }
        };
      }
      case ElementUnaryOp::SILU: {
        float beta = attrs.scalar.value_or(1.0f);
        return [beta](float x) -> float {
          return x / (1.0f + expf(-1.0f * beta * x));
        };
      }
      case ElementUnaryOp::SIN:
        return [](float x) -> float { return sinf(x); };
      case ElementUnaryOp::COS:
        return [](float x) -> float { return cosf(x); };
      case ElementUnaryOp::IDENTITY:
        return [](float x) -> float { return x; };
      case ElementUnaryOp::RSQRT:
        return [](float x) -> float { return 1.0f / sqrtf(x); };
      case ElementUnaryOp::SCALAR_MULTIPLY: {
        float scalar = attrs.scalar.value();
        return [scalar](float x) -> float { return x * scalar; };
      }
      case ElementUnaryOp::SCALAR_ADD: {
        float scalar = attrs.scalar.value();
        return [scalar](float x) -> float { return x + scalar; };
      }
      case ElementUnaryOp::SCALAR_SUB: {
        float scalar = attrs.scalar.value();
        return [scalar](float x) -> float { return x - scalar; };
      }
      case ElementUnaryOp::SCALAR_TRUE_DIV: {
        float scalar = attrs.scalar.value();
        return [scalar](float x) -> float { return x / scalar; };
      }
      case ElementUnaryOp::POW: {
        float scalar = attrs.scalar.value();
        return [scalar](float x) -> float { return powf(x, scalar); };
      }
      default:
        PANIC("Unhandled ElementUnaryOp {}", attrs.op);
    }
  }();

  return map_tensor_accessor_to(
      /*input=*/input,
      /*f=*/element_function,
      /*output=*/output);
}

void element_unary_cpu_backward_kernel(
    ElementUnaryAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad) {
  std::function<float(float, float)> element_function =
      [&]() -> std::function<float(float, float)> {
    switch (attrs.op) {
      case ElementUnaryOp::RELU:
        return [](float x, float fx) -> float {
          if (x > 0) {
            return 1;
          } else {
            return 0;
          }
        };
      case ElementUnaryOp::SIGMOID:
        return [](float x, float fx) -> float { return fx * (1.0f - fx); };
      case ElementUnaryOp::TANH:
        return [](float x, float fx) -> float { return 1 - fx * fx; };
      case ElementUnaryOp::GELU:
        return [](float x, float fx) -> float {
          float pdf_x = (1 / sqrtf(2 * M_PI)) * expf(-0.5f * x * x);
          return x * pdf_x + erfc(x);
        };
      case ElementUnaryOp::ELU: {
        float alpha = attrs.scalar.value();
        return [alpha](float x, float fx) -> float {
          if (x > 0) {
            return 1.0f;
          } else {
            return alpha * expf(x);
          }
        };
      }
      case ElementUnaryOp::SILU: {
        float beta = attrs.scalar.value_or(1.0f);
        return [beta](float x, float fx) -> float {
          float e_to_bx = expf(beta * x);
          return (e_to_bx * (beta * x + e_to_bx + 1.0f)) /
                 ((e_to_bx + 1) * (e_to_bx + 1));
        };
      }
      case ElementUnaryOp::SIN:
        return [](float x, float fx) -> float { return cosf(x); };
      case ElementUnaryOp::COS:
        return [](float x, float fx) -> float { return (-1.0f) * sinf(x); };
      case ElementUnaryOp::IDENTITY:
        return [](float x, float fx) -> float { return 1.0f; };
      case ElementUnaryOp::RSQRT:
        return [](float x, float fx) -> float { return -0.5f * fx * fx * fx; };
      case ElementUnaryOp::SCALAR_MULTIPLY: {
        float scalar = attrs.scalar.value();
        return [scalar](float x, float fx) -> float { return scalar; };
      }
      case ElementUnaryOp::SCALAR_ADD: {
        return [](float x, float fx) -> float { return 1.0f; };
      }
      case ElementUnaryOp::SCALAR_SUB: {
        return [](float x, float fx) -> float { return 1.0f; };
      }
      case ElementUnaryOp::SCALAR_TRUE_DIV: {
        float scalar = attrs.scalar.value();
        return [scalar](float x, float fx) -> float { return 1.0f / scalar; };
      }
      case ElementUnaryOp::POW: {
        float scalar = attrs.scalar.value();
        return [scalar](float x, float fx) -> float {
          return scalar * powf(x, scalar - 1.0f);
        };
      }
      default:
        PANIC("Unhandled ElementUnaryOp {}", attrs.op);
    }
  }();

  Allocator cpu_allocator = create_local_cpu_memory_allocator();

  GenericTensorAccessorR df_dx =
      read_only_accessor_from_write_accessor(map_tensor_accessors2(
          /*lhs=*/input,
          /*rhs=*/output,
          /*output_data_type=*/
          require_same(input.shape.data_type, input_grad.shape.data_type),
          /*f=*/element_function,
          /*output_allocator=*/cpu_allocator));

  tensor_accessor_elementwise_multiply_to(output_grad, df_dx, input_grad);
}

} // namespace FlexFlow
