#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CONCAT_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CONCAT_KERNELS_CPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"

namespace FlexFlow {

void concat_cpu_forward_kernel(
    GenericTensorAccessorW const &output,
    std::vector<GenericTensorAccessorR> const &inputs,
    ff_dim_t axis);

void concat_cpu_backward_kernel(
    GenericTensorAccessorR const &output_grad,
    std::vector<GenericTensorAccessorW> const &input_grads,
    ff_dim_t axis);

} // namespace FlexFlow

#endif
