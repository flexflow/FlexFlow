#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_TENSOR_ACCESSOR_BINARY_OPS_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_TENSOR_ACCESSOR_BINARY_OPS_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"

namespace FlexFlow {

GenericTensorAccessorW
    tensor_accessor_elementwise_add(GenericTensorAccessorR const &lhs,
                                    GenericTensorAccessorR const &rhs,
                                    Allocator &output_allocator);

void tensor_accessor_elementwise_add_to(GenericTensorAccessorR const &lhs,
                                        GenericTensorAccessorR const &rhs,
                                        GenericTensorAccessorW const &output);

void tensor_accessor_elementwise_add_to_inplace(
    GenericTensorAccessorR const &lhs, GenericTensorAccessorW const &output);

GenericTensorAccessorW
    tensor_accessor_elementwise_subtract(GenericTensorAccessorR const &lhs,
                                         GenericTensorAccessorR const &rhs,
                                         Allocator &output_allocator);

void tensor_accessor_elementwise_subtract_to(
    GenericTensorAccessorR const &lhs,
    GenericTensorAccessorR const &rhs,
    GenericTensorAccessorW const &output);

GenericTensorAccessorW
    tensor_accessor_elementwise_multiply(GenericTensorAccessorR const &lhs,
                                         GenericTensorAccessorR const &rhs,
                                         Allocator &output_allocator);

void tensor_accessor_elementwise_multiply_to(
    GenericTensorAccessorR const &lhs,
    GenericTensorAccessorR const &rhs,
    GenericTensorAccessorW const &output);

GenericTensorAccessorW tensor_accessor_matmul(GenericTensorAccessorR const &lhs,
                                              GenericTensorAccessorR const &rhs,
                                              Allocator &output_allocator);

void tensor_accessor_matmul_to(GenericTensorAccessorR const &lhs,
                               GenericTensorAccessorR const &rhs,
                               GenericTensorAccessorW const &output);

GenericTensorAccessorW
    tensor_accessor_batch_matmul(GenericTensorAccessorR const &lhs,
                                 GenericTensorAccessorR const &rhs,
                                 Allocator &output_allocator);

void tensor_accessor_batch_matmul_to(GenericTensorAccessorR const &lhs,
                                     GenericTensorAccessorR const &rhs,
                                     GenericTensorAccessorW const &output);

} // namespace FlexFlow

#endif
