#include "kernels/reshape_kernels_cpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

void reshape_cpu_forward_kernel(GenericTensorAccessorR const &input,
                                GenericTensorAccessorW const &output) {
  copy_accessor_data_to_l_from_r(output, input);
}

void reshape_cpu_backward_kernel(GenericTensorAccessorR const &output_grad,
                                 GenericTensorAccessorW const &input_grad) {
  copy_accessor_data_to_l_from_r(input_grad, output_grad);
}

} // namespace FlexFlow
