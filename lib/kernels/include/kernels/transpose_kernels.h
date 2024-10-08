#ifndef _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H

#include "device.h"
#include "kernels/accessor.h"
#include <vector>

namespace FlexFlow {

struct TransposePerDeviceState {
  int num_dim;
  req<std::vector<legion_dim_t>> perm;
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(TransposePerDeviceState,
                                             num_dim,
                                             perm);

namespace Kernels::Transpose {

TransposePerDeviceState init_kernel(int num_dim,
                                    std::vector<ff_dim_t> const &perm);

void forward_kernel(cudaStream_t stream,
                    TransposePerDeviceState const &m,
                    GenericTensorAccessorR const &input,
                    GenericTensorAccessorW const &output);

void backward_kernel(cudaStream_t stream,
                     TransposePerDeviceState const &m,
                     GenericTensorAccessorW const &in_grad,
                     GenericTensorAccessorR const &out_grad);

} // namespace Kernels::Transpose
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H
