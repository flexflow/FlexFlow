#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ACCESSORS_ARE_WITHIN_EPSILON_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ACCESSORS_ARE_WITHIN_EPSILON_H

#include "kernels/accessor.h"

namespace FlexFlow {

bool accessors_are_within_epsilon(GenericTensorAccessorR const &,
                                  GenericTensorAccessorR const &,
                                  std::optional<float> epsilon = std::nullopt);

} // namespace FlexFlow

#endif
