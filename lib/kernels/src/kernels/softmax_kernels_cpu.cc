#include "kernels/softmax_kernels_cpu.h"
#include "utils/exception.h"

namespace FlexFlow {

void softmax_cpu_forward_kernel(SoftmaxAttrs const &attrs,
                                GenericTensorAccessorR const &input,
                                GenericTensorAccessorW const &output) {
  NOT_IMPLEMENTED();
}

void softmax_cpu_backward_kernel(SoftmaxAttrs const &attrs,
                                 GenericTensorAccessorR const &output,
                                 GenericTensorAccessorR const &output_grad,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &input_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
