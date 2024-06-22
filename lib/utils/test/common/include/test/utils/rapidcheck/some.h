#ifndef _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_RAPIDCHECK_SOME_H
#define _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_RAPIDCHECK_SOME_H

#include <rapidcheck.h>

namespace FlexFlow {

template <typename T>
T some() {
  rc::Random r{};
  return rc::gen::arbitrary<T>()(r).value();
}

} // namespace FlexFlow

#endif
