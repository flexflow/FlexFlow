#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_FLAT_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_FLAT_KERNELS_CPU_H

#include "kernels/accessor.h"

namespace FlexFlow {

void flat_cpu_forward_kernel(GenericTensorAccessorR const &input,
                             GenericTensorAccessorW const &output);

void flat_cpu_backward_kernel(GenericTensorAccessorR const &output_grad,
                              GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow

#endif
