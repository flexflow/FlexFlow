#ifndef _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "kernels/per_device_op_state.h"

namespace FlexFlow {

class CombinePerDeviceState : public PerDeviceOpState {
public:
  CombinePerDeviceState(FFHandler handle);
  DataType data_type;
};

namespace Kernels {
namespace Combine {

void forward_kernel(ffStream_t stream, CombinePerDeviceState const *m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream, CombinePerDeviceState const *m,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const &input_grad);

} // namespace Combine
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_H
