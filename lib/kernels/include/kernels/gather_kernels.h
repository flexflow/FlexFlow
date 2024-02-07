#ifndef _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H

#include "accessor.h"
#include "kernels/device.h"

namespace FlexFlow {

struct GatherPerDeviceState {
   PerDeviceFFHandle handle;
   int legion_dim;
   DataType index_data_type;
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(GatherPerDeviceState, handle, index_data_type, legion_dim);

namespace Kernels {
namespace Gather {

 GatherPerDeviceState init_kernel(PerDeviceFFHandle handle, int legion_dim, DataType index_data_type);

void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorR const &index,
                    GenericTensorAccessorW const &output,
                    size_t stride,
                    ff_dim_t dim);
void backward_kernel(ffStream_t stream,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &index,
                     GenericTensorAccessorW const &input_grad,
                     size_t stride,
                     ff_dim_t dim);
} // namespace Gather
} // namespace Kernels
} // namespace FlexFlow

#endif
