#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_POOL_2D_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_POOL_2D_KERNELS_CPU_H

#include "kernels/accessor.h"
#include "op-attrs/ops/pool_2d_attrs.dtg.h"

namespace FlexFlow {

void pool_2d_cpu_forward_kernel(Pool2DAttrs const &attrs,
                                GenericTensorAccessorR const &input,
                                GenericTensorAccessorW const &output);

void pool_2d_cpu_backward_kernel(Pool2DAttrs const &attrs,
                                 GenericTensorAccessorR const &output,
                                 GenericTensorAccessorR const &output_grad,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow

#endif
