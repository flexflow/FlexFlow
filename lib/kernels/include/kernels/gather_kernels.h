#ifndef _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H

#include "accessor.h"
#include "kernels/device.h"

#include "kernels/per_device_op_state.h"

namespace FlexFlow {

class Gather;

class GatherPerDeviceState : public PerDeviceOpState {
public:
  GatherPerDeviceState(FFHandler handler, Gather const *gather);

public:
  int legion_dim;
};

namespace Kernels {
namespace Gather {
void forward_kernel_wrapper(GatherPerDeviceState const *m,
                            GenericTensorAccessorR const &input,
                            GenericTensorAccessorR const &index,
                            GenericTensorAccessorW const &output);
void backward_kernel_wrapper(GatherPerDeviceState const *m,
                             GenericTensorAccessorR const &output_grad,
                             GenericTensorAccessorR const &index,
                             GenericTensorAccessorW const &input_grad);
namespace Internal {
template <typename IndexType>
void forward_kernel(float const *input_ptr,
                    IndexType const *index_ptr,
                    float *output_ptr,
                    Legion::coord_t output_size,
                    Legion::coord_t stride,
                    Legion::coord_t dim_size,
                    ffStream_t stream);
template <typename IndexType>
void backward_kernel(float const *output_grad_ptr,
                     IndexType const *index_ptr,
                     float *input_grad_ptr,
                     Legion::coord_t output_size,
                     Legion::coord_t stride,
                     Legion::coord_t dim_size,
                     ffStream_t stream);
} // namespace Internal
} // namespace Gather
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_GATHER_KERNELS_H
