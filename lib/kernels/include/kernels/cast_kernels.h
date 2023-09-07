#ifndef _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_CAST_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device.h"

namespace FlexFlow {

struct CastPerDeviceState {
  PerDeviceFFHandle handle;
  DataType input_data_type;
  req<DataType> output_data_type;
};

FF_VISITABLE_STRUCT_NO_EQ(CastPerDeviceState,
                          handle,
                          input_data_type,
                          output_data_type);

namespace Kernels {
namespace Cast {

CastPerDeviceState
    init_kernel(PerDeviceFFHandle const &, DataType input, DataType output);

void forward_kernel(ffStream_t stream,
                    CastPerDeviceState const *,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(ffStream_t stream,
                     CastPerDeviceState const *,
                     GenericTensorAccessorR const &input,
                     GenericTensorAccessorW const &output);

} // namespace Cast
} // namespace Kernels
} // namespace FlexFlow

#endif