#ifndef _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CONCAT_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "kernels/per_device_op_state.h"

namespace FlexFlow {

class ConcatPerDeviceState : public PerDeviceOpState {
public:
  ConcatPerDeviceState(FFHandler handle) : PerDeviceOpState(handle){};
  int legion_axis;
  char op_name[MAX_OPNAME];
};

namespace Kernels {
namespace Concat {

void init_meta(ConcatPerDeviceState *meta, int legion_axis);

void forward_kernel(ffStream_t stream, ConcatPerDeviceState const *m,
                    GenericTensorAccessorW const &output,
                    GenericTensorAccessorR const *inputs, int num_inputs);

void backward_kernel(ffStream_t stream, ConcatPerDeviceState const *m,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const *input_grads, int num_inputs);

} // namespace Concat
} // namespace Kernels
} // namespace FlexFlow

#endif
