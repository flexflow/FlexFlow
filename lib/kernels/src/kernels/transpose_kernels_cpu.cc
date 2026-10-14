#include "kernels/transpose_kernels_cpu.h"
#include "kernels/tensor_accessor_unary_ops.h"

namespace FlexFlow {

void transpose_cpu_forward_kernel(TransposeAttrs const &attrs,
                                  GenericTensorAccessorR const &input,
                                  GenericTensorAccessorW const &output) {
  return tensor_accessor_transpose_to(input, attrs.permutation, output);
}

void transpose_cpu_backward_kernel(TransposeAttrs const &attrs,
                                   GenericTensorAccessorR const &out_grad,
                                   GenericTensorAccessorW const &in_grad) {
  return tensor_accessor_transpose_to(
      out_grad, invert_tensor_dim_permutation(attrs.permutation), in_grad);
}

} // namespace FlexFlow
