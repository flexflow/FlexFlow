#ifndef _FLEXFLOW_LIB_UTILS_RAPIDCHECK_EXTRA_INCLUDE_UTILS_RAPIDCHECK_EXTRA_SOME_H
#define _FLEXFLOW_LIB_UTILS_RAPIDCHECK_EXTRA_INCLUDE_UTILS_RAPIDCHECK_EXTRA_SOME_H

#include "rapidcheck.h"

namespace FlexFlow {

template <typename T>
T some() {
  rc::Random r{};
  return rc::gen::arbitrary<T>()(r).value();
}

} // namespace FlexFlow

#endif
