#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CREATE_RANDOM_FILLED_ACCESSOR_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CREATE_RANDOM_FILLED_ACCESSOR_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "op-attrs/tensor_shape.dtg.h"
#include <random>

namespace FlexFlow {

GenericTensorAccessorW create_random_filled_accessor_w(TensorShape const &shape,
                                                       Allocator &allocator,
                                                       int seed = 0);

GenericTensorAccessorR create_random_filled_accessor_r(TensorShape const &shape,
                                                       Allocator &allocator,
                                                       int seed = 0);

GenericTensorAccessorW create_random_filled_accessor_w_with_gen(
    TensorShape const &shape, Allocator &allocator, std::mt19937 &gen);

GenericTensorAccessorR create_random_filled_accessor_r_with_gen(
    TensorShape const &shape, Allocator &allocator, std::mt19937 &gen);

} // namespace FlexFlow

#endif
