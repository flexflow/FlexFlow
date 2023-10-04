#ifndef _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "op-attrs/activation.h"

namespace FlexFlow {
namespace Kernels {
namespace Cast {

void forward_kernel(ffStream_t stream,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output,
                    DataType input_type,
                    DataType output_type,
                    PerDeviceFFHandle handle);

void backward_kernel(ffStream_t stream,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &output,
                     DataType input_type,
                     DataType output_type,
                     PerDeviceFFHandle handle);

} // namespace Cast
} // namespace Kernels
} // namespace FlexFlow

#endif
