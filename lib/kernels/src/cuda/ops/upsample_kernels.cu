#include "internal/device.h"
#include "kernels/upsample_kernels_gpu.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

void upsample_gpu_forward_kernel(ffStream_t stream,
                                 UpsampleAttrs const &attrs,
                                 GenericTensorAccessorR const &input,
                                 GenericTensorAccessorW const &output) {
  NOT_IMPLEMENTED();
}

void upsample_gpu_backward_kernel(ffStream_t stream,
                                  UpsampleAttrs const &attrs,
                                  GenericTensorAccessorR const &output,
                                  GenericTensorAccessorR const &output_grad,
                                  GenericTensorAccessorR const &input,
                                  GenericTensorAccessorW const &input_grad) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
