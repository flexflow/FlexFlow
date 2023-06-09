#ifndef _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H

#include "kernels/accessor.h"
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

void forward_kernel(ffStream_t stream,
                    ReshapePerDeviceState const *m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     ReshapePerDeviceState const *m,
                     GenericTensorAccessorW const &input,
                     GenericTensorAccessorR const &output);

} // namespace Reshape
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H