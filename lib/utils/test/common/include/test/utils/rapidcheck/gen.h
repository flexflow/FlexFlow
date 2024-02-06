#ifndef _FLEXFLOW_UTILS_LIB_TEST_COMMON_INCLUDE_UTILS_TEST_RAPIDCHECK_GEN_H
#define _FLEXFLOW_UTILS_LIB_TEST_COMMON_INCLUDE_UTILS_TEST_RAPIDCHECK_GEN_H

#include "rapidcheck.h"
#include <unordered_set>

namespace rc {

template <typename C, typename T = typename C::value_type>
Gen<std::unordered_set<T>> subset_of(C const &sets) {
  return gen::exec([&] {
    std::unordered_set<T> result;
    for (auto const &elem : sets) {
      if (*gen::arbitrary<bool>()) {
        result.insert(elem);
      }
    }
    return result;
  });
}

} // namespace rc

#endif
