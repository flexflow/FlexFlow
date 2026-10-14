#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CREATE_CONSTANT_FILLED_ACCESSOR_R_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_CREATE_CONSTANT_FILLED_ACCESSOR_R_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "op-attrs/datatype_value.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

GenericTensorAccessorW create_constant_filled_accessor_w(
    TensorShape const &shape, Allocator &allocator, DataTypeValue val);

GenericTensorAccessorR create_constant_filled_accessor_r(
    TensorShape const &shape, Allocator &allocator, DataTypeValue val);

} // namespace FlexFlow

#endif
