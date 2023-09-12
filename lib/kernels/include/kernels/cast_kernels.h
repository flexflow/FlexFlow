#ifndef _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "op-attrs/activation.h"

namespace FlexFlow {

struct CastPerDeviceState {
  PerDeviceFFHandle handle;
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(CastPerDeviceState, handle);

namespace Kernels {
namespace Cast {

CastPerDeviceState init_kernel(PerDeviceFFHandle const &handle);

void forward_kernel(ffStream_t stream,
                    CastPerDeviceState const *m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    DataType input_type,
                    DataType output_type);

void backward_kernel(ffStream_t stream,
                     CastPerDeviceState const *m,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &output,
                     DataType input_type,
                     DataType output_type);

} // namespace Cast
} // namespace Kernels
} // namespace FlexFlow

#endif
