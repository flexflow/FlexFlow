#ifndef _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H

#include "kernels/device.h"

#include "kernels/per_device_op_state.h"

namespace FlexFlow {

class RepartitionPerDeviceState : public PerDeviceOpState {
public:
  RepartitionPerDeviceState(FFHandler handle);
  DataType data_type;
};

namespace Kernels {
namespace Repartition {

template <typename T>
void forward_kernel(T const *input_ptr, T *output_ptr, size_t num_elements);

template <typename T>
void backward_kernel(T const *output_grad_ptr,
                     T *input_grad_ptr,
                     size_t num_elements);

} // namespace Repartition
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H
