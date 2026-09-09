#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_NORM_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_BATCH_NORM_KERNELS_CPU_H

#include "kernels/accessor.h"
#include "op-attrs/ops/batch_norm_attrs.dtg.h"

namespace FlexFlow {

void batch_norm_cpu_forward_kernel(BatchNormAttrs const &attrs,
                                   GenericTensorAccessorR const &input,
                                   GenericTensorAccessorR const &gamma,
                                   GenericTensorAccessorR const &beta,
                                   GenericTensorAccessorW const &output);

void batch_norm_cpu_backward_kernel(BatchNormAttrs const &attrs,
                                    GenericTensorAccessorR const &output,
                                    GenericTensorAccessorR const &output_grad,
                                    GenericTensorAccessorR const &input,
                                    GenericTensorAccessorW const &input_grad,
                                    GenericTensorAccessorR const &gamma,
                                    GenericTensorAccessorW const &gamma_grad,
                                    GenericTensorAccessorW const &beta_grad);

} // namespace FlexFlow

#endif
