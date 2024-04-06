#ifndef _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H
#define _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H

#include "device.h"
#include <vector>

namespace FlexFlow {

struct TransposePerDeviceState {
  int num_dim;
  req<int> perm[MAX_TENSOR_DIM];
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(TransposePerDeviceState,
                                             num_dim,
                                             perm);

namespace Kernels {
namespace Transpose {

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

// namespace Internal {

// void forward_kernel_wrapper(TransposePerDeviceState const &m,
//                             float const *input_ptr,
//                             float *output_ptr,
//                             GenericTensorAccessorR const &input,
//                             GenericTensorAccessorW const &output,
//                             ffStream_t stream);
// void backward_kernel_wrapper(TransposePerDeviceState const &m,
//                              float *input_grad_ptr,
//                              float const *output_grad_ptr,
//                              GenericTensorAccessorW const &in_grad,
//                              GenericTensorAccessorR const &out_grad,
//                              ffStream_t stream);

// } // namespace Internal
} // namespace Transpose
} // namespace Kernels
} // namespace FlexFlow

#endif // _FLEXFLOW_OPS_KERNELS_TRANSPOSE_KERNELS_H
