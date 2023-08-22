#ifndef _FLEXFLOW_OPS_KERNELS_FLAT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_FLAT_KERNELS_H

#include "kernels/device.h"

namespace FlexFlow {

class FlatPerDeviceState : public PerDeviceOpState {
public:
  FlatPerDeviceState(FFHandler handle) : PerDeviceOpState(handle){};
};

namespace Kernels {
namespace Flat {

void forward_kernel(ffStream_t stream,
                    float const *input_ptr,
                    float *output_ptr,
                    size_t num_elements);
void backward_kernel(ffStream_t stream,
                     float *input_grad_ptr,
                     float const *output_grad_ptr,
                     size_t num_elements);

} // namespace Flat
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_FLAT_KERNELS_H
