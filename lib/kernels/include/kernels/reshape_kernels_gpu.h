#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_RESHAPE_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_RESHAPE_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"

namespace FlexFlow {

void reshape_gpu_forward_kernel(ffStream_t stream,
                                GenericTensorAccessorR const &input,
                                GenericTensorAccessorW const &output);

void reshape_gpu_backward_kernel(ffStream_t stream,
                                 GenericTensorAccessorR const &output,
                                 GenericTensorAccessorW const &input);

} // namespace FlexFlow

#endif
