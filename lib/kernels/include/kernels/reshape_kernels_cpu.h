#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_RESHAPE_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_RESHAPE_KERNELS_CPU_H

#include "kernels/accessor.h"

namespace FlexFlow {

void reshape_cpu_forward_kernel(GenericTensorAccessorR const &input,
                                GenericTensorAccessorW const &output);

void reshape_cpu_backward_kernel(GenericTensorAccessorR const &output,
                                 GenericTensorAccessorR const &output_grad,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow

#endif
