#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONNEGATIVE_INT_NUM_ELEMENTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NONNEGATIVE_INT_NUM_ELEMENTS_H

#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

template <typename T>
nonnegative_int num_elements(T const &t) {
  size_t t_size = t.size();
  return nonnegative_int{t_size};
}

} // namespace FlexFlow

#endif
