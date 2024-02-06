#ifndef _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "utils/required_core.h"

namespace FlexFlow {

struct ReshapePerDeviceState {
  req<DataType> data_type;
};

FF_VISITABLE_STRUCT(ReshapePerDeviceState, data_type);

namespace Kernels {
namespace Reshape {

ReshapePerDeviceState init_kernel(DataType data_type);

void forward_kernel(ffStream_t stream,
                    ReshapePerDeviceState const &per_device_state,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     ReshapePerDeviceState const &per_device_state,
                     GenericTensorAccessorW const &input,
                     GenericTensorAccessorR const &output);

} // namespace Reshape
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H
