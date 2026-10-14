#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_TENSOR_ACCESSOR_SPLIT_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_TENSOR_ACCESSOR_SPLIT_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "op-attrs/ff_dim_t.dtg.h"
#include <vector>

namespace FlexFlow {

std::vector<GenericTensorAccessorW>
    tensor_accessor_split(GenericTensorAccessorR const &input,
                          ff_dim_t const &axis,
                          std::vector<positive_int> const &sizes,
                          Allocator &allocator);

void tensor_accessor_split_to(
    GenericTensorAccessorR const &input,
    ff_dim_t const &axis,
    std::vector<positive_int> const &sizes,
    std::vector<GenericTensorAccessorW> const &outputs);

} // namespace FlexFlow

#endif
