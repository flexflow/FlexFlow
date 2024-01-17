#ifndef _FLEXFLOW_UTILS_LIB_TEST_COMMON_INCLUDE_UTILS_TEST_RAPIDCHECK_GEN_H
#define _FLEXFLOW_UTILS_LIB_TEST_COMMON_INCLUDE_UTILS_TEST_RAPIDCHECK_GEN_H

#include "rapidcheck.h"
#include <unordered_set>

namespace rc {

template <typename C, typename T = typename C::value_type>
Gen<std::unordered_set<T>> subset_of(C const &sets) {
  return gen::exec([&] {
    auto s = sets;
    int subset_size = *gen::inRange(0, (int)s.size());
    auto idxs = *gen::unique<std::vector<int>>(subset_size,
                                               gen::inRange(0, (int)s.size()));
    std::unordered_set<T> result;
    for (int idx : idxs) {
      result.insert(*(sets.cbegin() + idx));
    }
    return result;
  });
}

} // namespace rc

#endif
