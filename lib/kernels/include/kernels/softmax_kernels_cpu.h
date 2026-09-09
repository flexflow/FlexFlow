#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SOFTMAX_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_SOFTMAX_KERNELS_CPU_H

#include "kernels/accessor.h"
#include "op-attrs/ops/softmax_attrs.dtg.h"

namespace FlexFlow {

void softmax_cpu_forward_kernel(SoftmaxAttrs const &attrs,
                                GenericTensorAccessorR const &input,
                                GenericTensorAccessorW const &output);

void softmax_cpu_backward_kernel(SoftmaxAttrs const &attrs,
                                 GenericTensorAccessorR const &output,
                                 GenericTensorAccessorR const &output_grad,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow

#endif
