#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_FLAT_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_FLAT_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"

namespace FlexFlow {

void flat_gpu_forward_kernel(ffStream_t stream,
                             void const *input_ptr,
                             void *output_ptr,
                             size_t num_elements,
                             size_t element_size_in_bytes);

void flat_gpu_backward_kernel(ffStream_t stream,
                              float const *output_grad_ptr,
                              float *input_grad_ptr,
                              size_t num_elements);

} // namespace FlexFlow

#endif
