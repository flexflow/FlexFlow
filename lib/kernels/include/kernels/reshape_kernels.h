#ifndef _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H

#include "kernels/device.h"
#include "kernels/per_device_op_state.h"

namespace FlexFlow {

class ReshapePerDeviceState : public PerDeviceOpState {
public:
  ReshapePerDeviceState(FFHandler handler);
  DataType data_type;
};

namespace Kernels {
namespace Reshape {

template <typename T>
void forward_kernel(ffStream_t stream,
                    T const *input_ptr,
                    T *output_ptr,
                    size_t num_elements);
template <typename T>
void backward_kernel(ffStream_t stream,
                     T *input_grad_ptr,
                     T const *output_grad_ptr,
                     size_t num_elements);


} // namespace Reshape
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H