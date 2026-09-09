#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_UPSAMPLE_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_UPSAMPLE_KERNELS_CPU_H

#include "kernels/accessor.h"
#include "op-attrs/ops/upsample_attrs.dtg.h"

namespace FlexFlow {

void upsample_cpu_forward_kernel(UpsampleAttrs const &attrs,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &output);

void upsample_cpu_backward_kernel(UpsampleAttrs const &attrs,
                                  GenericTensorAccessorR const &output,
                                  GenericTensorAccessorR const &output_grad,
                                  GenericTensorAccessorR const &input,
                                  GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow

#endif
