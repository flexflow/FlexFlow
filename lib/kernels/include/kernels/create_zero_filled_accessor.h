#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CREATE_ZERO_FILLED_ACCESSOR_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CREATE_ZERO_FILLED_ACCESSOR_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

GenericTensorAccessorW create_zero_filled_accessor_w(TensorShape const &shape,
                                                     Allocator &allocator);

GenericTensorAccessorR create_zero_filled_accessor_r(TensorShape const &shape,
                                                     Allocator &allocator);

} // namespace FlexFlow

#endif
