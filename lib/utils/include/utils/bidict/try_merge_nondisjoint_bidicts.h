#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_TRY_MERGE_NONDISJOINT_BIDICTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_TRY_MERGE_NONDISJOINT_BIDICTS_H

#include "utils/bidict/bidict.h"

namespace FlexFlow {

template <typename L, typename R>
std::optional<bidict<L, R>>
    try_merge_nondisjoint_bidicts(bidict<L, R> d1, bidict<L, R> const &d2) {
  for (auto const &[l, r] : d2) {
    bool are_now_equated = d1.try_equate(l, r);
    if (!are_now_equated) {
      return std::nullopt;
    }
  }

  return d1;
}

} // namespace FlexFlow

#endif
