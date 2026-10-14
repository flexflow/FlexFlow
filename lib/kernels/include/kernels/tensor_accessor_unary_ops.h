#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_TENSOR_ACCESSOR_UNARY_OPS_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_TENSOR_ACCESSOR_UNARY_OPS_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "op-attrs/tensor_dim_permutation.h"

namespace FlexFlow {

GenericTensorAccessorW tensor_accessor_exp(GenericTensorAccessorR const &input,
                                           Allocator &output_allocator);

void tensor_accessor_exp_to(GenericTensorAccessorR const &input,
                            GenericTensorAccessorW const &output);

void tensor_accessor_exp_inplace(GenericTensorAccessorW const &input);

GenericTensorAccessorW
    tensor_accessor_scale_by_constant(GenericTensorAccessorR const &input,
                                      float constant,
                                      Allocator &output_allocator);

void tensor_accessor_scale_by_constant_inplace(
    GenericTensorAccessorW const &input, float constant);

GenericTensorAccessorW tensor_accessor_relu(GenericTensorAccessorR const &input,
                                            Allocator &output_allocator);

void tensor_accessor_relu_to(GenericTensorAccessorR const &input,
                             GenericTensorAccessorW const &output);

GenericTensorAccessorW
    tensor_accessor_broadcast(GenericTensorAccessorR const &input,
                              TensorDims const &output_dims,
                              Allocator &output_allocator);

void tensor_accessor_broadcast_to(GenericTensorAccessorR const &input,
                                  TensorDims const &output_dims,
                                  GenericTensorAccessorW const &output);

GenericTensorAccessorW
    tensor_accessor_transpose(GenericTensorAccessorR const &input,
                              TensorDimPermutation const &dim_permutation,
                              Allocator &output_allocator);

void tensor_accessor_transpose_to(GenericTensorAccessorR const &input,
                                  TensorDimPermutation const &dim_permutation,
                                  GenericTensorAccessorW const &output);

GenericTensorAccessorW
    tensor_accessor_transpose_2d(GenericTensorAccessorR const &input,
                                 Allocator &output_allocator);

void tensor_accessor_transpose_2d_to(GenericTensorAccessorR const &input,
                                     GenericTensorAccessorW const &output);

GenericTensorAccessorW
    tensor_accessor_batch_transpose(GenericTensorAccessorR const &input,
                                    Allocator &output_allocator);

void tensor_accessor_batch_transpose_to(GenericTensorAccessorR const &input,
                                        GenericTensorAccessorW const &output);

GenericTensorAccessorW
    tensor_accessor_reduce(GenericTensorAccessorR const &input,
                           ff_dim_t dim,
                           Allocator &output_allocator);

void tensor_accessor_reduce_to(GenericTensorAccessorR const &input,
                               ff_dim_t dim,
                               GenericTensorAccessorW const &output);

} // namespace FlexFlow

#endif
