#ifndef _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H

#include "accessor.h"
#include "kernels/device.h"

namespace FlexFlow {

struct GatherPerDeviceState {
  PerDeviceFFHandle handle;
  legion_dim_t legion_dim;
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(GatherPerDeviceState,
                                             handle,
                                             legion_dim);

namespace Kernels::Gather {

void forward_kernel(ffStream_t stream,
                    GatherPerDeviceState const &m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorR const &index,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     GatherPerDeviceState const &m,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &index,
                     GenericTensorAccessorW const &input_grad);

} // namespace Kernels::Gather
} // namespace FlexFlow

#endif
