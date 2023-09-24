#ifndef _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"

namespace FlexFlow {

struct RepartitionPerDeviceState {
  PerDeviceFFHandle handle;
  DataType data_type;
};

FF_VISITABLE_STRUCT_NO_EQ(RepartitionPerDeviceState, handle, data_type);

namespace Kernels {
namespace Repartition {

RepartitionPerDeviceState init_kernel(PerDeviceFFHandle const &handle,
                                      DataType data_type);

void forward_kernel(ffStream_t stream,
                    RepartitionPerDeviceState const &m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     RepartitionPerDeviceState const &m,
                     GenericTensorAccessorW const &output_grad,
                     GenericTensorAccessorR const &input_grad);

} // namespace Repartition
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_PARTITION_KERNELS_H
