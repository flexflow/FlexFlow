#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ACCESSORS_ARE_APPROXIMATELY_EQUAL_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ACCESSORS_ARE_APPROXIMATELY_EQUAL_H

#include "kernels/accessor.h"

namespace FlexFlow {

bool accessors_are_approximately_equal(
    GenericTensorAccessorR const &,
    GenericTensorAccessorR const &,
    std::optional<float> relative_error_allowed = std::nullopt);

} // namespace FlexFlow

#endif
