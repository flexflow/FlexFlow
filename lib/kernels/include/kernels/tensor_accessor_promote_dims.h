#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_TENSOR_ACCESSOR_PROMOTE_DIMS_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_TENSOR_ACCESSOR_PROMOTE_DIMS_H

#include "kernels/accessor.h"
#include "op-attrs/tensor_dims.dtg.h"

namespace FlexFlow {

GenericTensorAccessorR
    tensor_accessor_promote_dims(GenericTensorAccessorR const &,
                                 TensorDims const &);

} // namespace FlexFlow

#endif
