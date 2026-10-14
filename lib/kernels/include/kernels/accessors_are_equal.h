#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ACCESSORS_ARE_EQUAL_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ACCESSORS_ARE_EQUAL_H

#include "kernels/accessor.h"

namespace FlexFlow {

bool accessors_are_equal(GenericTensorAccessorR const &,
                         GenericTensorAccessorR const &);

} // namespace FlexFlow

#endif
