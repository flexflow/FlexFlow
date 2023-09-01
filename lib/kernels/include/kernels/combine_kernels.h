#ifndef _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"

namespace FlexFlow {

struct CombinePerDeviceState {
  req<DataType> data_type;
};

FF_VISITABLE_STRUCT_NO_EQ(CombinePerDeviceState, data_type);

namespace Kernels {
namespace Combine {

CombinePerDeviceState init_kernel(DataType data_type);

void forward_kernel(ffStream_t stream,
                    CombinePerDeviceState const *m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     CombinePerDeviceState const *m,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorW const &input_grad);

} // namespace Combine
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_COMBINE_KERNELS_H
