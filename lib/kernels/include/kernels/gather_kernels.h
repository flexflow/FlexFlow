#ifndef _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H

#include "accessor.h"
#include "kernels/device.h"

namespace FlexFlow {

class GatherPerDeviceState : public PerDeviceOpState {
public:
  GatherPerDeviceState(FFHandler handler);
  int legion_dim;
  DataType index_data_type;
};

namespace Kernels {
namespace Gather {
void forward_kernel(ffStream_t stream,
                    GatherPerDeviceState const *m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorR const &index,
                    GenericTensorAccessorW const &output,
                    size_t stride,
                    size_t input_dim_size,
                    size_t output_dim_size);
void backward_kernel(ffStream_t stream,
                     GatherPerDeviceState const *m,
                     GenericTensorAccessorR const &output_grad,
                     GenericTensorAccessorR const &index,
                     GenericTensorAccessorW const &input_grad,
                     size_t stride,
                     size_t input_dim_size,
                     size_t output_dim_size);
} // namespace Gather
} // namespace Kernels
} // namespace FlexFlow

#endif
