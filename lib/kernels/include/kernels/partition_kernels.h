#ifndef _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H

#include "kernels/accessor.h"
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

void forward_kernel(ffStream_t stream,
                    RepartitionPerDeviceState const *m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     RepartitionPerDeviceState const *m,
                     GenericTensorAccessorW const &output_grad,
                     GenericTensorAccessorR const &input_grad);

} // namespace Repartition
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H
