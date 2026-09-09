#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_TRANSPOSE_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_TRANSPOSE_KERNELS_CPU_H

#include "kernels/accessor.h"
#include "op-attrs/ops/transpose_attrs.dtg.h"

namespace FlexFlow {

void transpose_cpu_forward_kernel(TransposeAttrs const &attrs,
                                  GenericTensorAccessorR const &input,
                                  GenericTensorAccessorW const &output);

void transpose_cpu_backward_kernel(TransposeAttrs const &attrs,
                                   GenericTensorAccessorR const &output,
                                   GenericTensorAccessorR const &output_grad,
                                   GenericTensorAccessorR const &input,
                                   GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow

#endif
