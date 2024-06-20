#ifndef _FLEXFLOW_OPS_KERNELS_FLAT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_FLAT_KERNELS_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/op_meta.h"

namespace FlexFlow {

class Flat;

class FlatMeta : public OpMeta {
public:
  FlatMeta(FFHandler handle, Flat const *flat);
};

namespace Kernels {
namespace Flat {

void forward_kernel_wrapper(float const *input_ptr,
                            float *output_ptr,
                            size_t num_elements);
void backward_kernel_wrapper(float *input_grad_ptr,
                             float const *output_grad_ptr,
                             size_t num_elements);

namespace Internal {

void forward_kernel(float const *input_ptr,
                    float *output_ptr,
                    size_t num_elements,
                    ffStream_t stream);
void backward_kernel(float *input_grad_ptr,
                     float const *output_grad_ptr,
                     size_t num_elements,
                     ffStream_t stream);

} // namespace Internal
} // namespace Flat
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_FLAT_KERNELS_H
