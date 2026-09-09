#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_UPSAMPLE_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_UPSAMPLE_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "op-attrs/ops/upsample_attrs.dtg.h"

namespace FlexFlow {

void upsample_gpu_forward_kernel(ffStream_t stream,
                                 UpsampleAttrs const &attrs,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &output);

void upsample_gpu_backward_kernel(ffStream_t stream,
                                  UpsampleAttrs const &attrs,
                                  GenericTensorAccessorR const &output,
                                  GenericTensorAccessorR const &output_grad,
                                  GenericTensorAccessorR const &input,
                                  GenericTensorAccessorW const &input_grad);

} // namespace FlexFlow

#endif
