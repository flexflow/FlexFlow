#ifndef _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_MERGE_H
#define _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_MERGE_H

#include "utils/bidict/bidict.h"
#include "utils/ff_exceptions/ff_exceptions.h"
#include "fmt/format.h"

namespace FlexFlow {

template <typename L, typename R>
bidict<L, R> merge_maps(bidict<L, R> const &lhs, bidict<L, R> const &rhs) {
  bidict<L, R> result;
  for (auto const &[k, v] : lhs) {
    result.equate(k, v);
  }
  for (auto const &[k, v] : rhs) {
    if (result.find_l(k) != result.end()) {
      throw mk_runtime_error(fmt::format("Refusing to merge non-disjoint maps! Found overlapping key/left {}", k));
    }
    if (result.find_r(v) != result.end()) {
      throw mk_runtime_error(fmt::format("Refusing to merge non-disjoint maps! Found overlapping value/right {}", v));
    }
    result.equate(k, v);
  }

  return result;
}

} // namespace FlexFlow

#endif
