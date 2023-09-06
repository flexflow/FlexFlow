#ifndef _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H

#include "attention_kernels.h"
#include "datatype_dispatch.h"
#include "kernels/accessor.h"
#include "kernels/device.h"
#include "utils/required_core.h"

namespace FlexFlow {

struct ReshapePerDeviceState  {
  req<DataType> data_type;
};

FF_VISITABLE_STRUCT_NO_EQ(ReshapePerDeviceState, data_type);

ReshapePerDeviceState init_kernel(DataType data_type);

namespace Kernels {
namespace Reshape {

void forward_kernel(ffStream_t stream,
                    ReshapePerDeviceState const & meta,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     ReshapePerDeviceState const & meta,
                     GenericTensorAccessorW const &input,
                     GenericTensorAccessorR const &output);

} // namespace Reshape
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_RESHAPE_KERNELS_H
