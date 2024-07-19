#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_TRY_MERGE_NONDISJOINT_BIDICTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_TRY_MERGE_NONDISJOINT_BIDICTS_H

#include "utils/bidict/bidict.h"

namespace FlexFlow {

template <typename L, typename R>
std::optional<bidict<L, R>>
    try_merge_nondisjoint_bidicts(bidict<L, R> const &d1,
                                  bidict<L, R> const &d2) {
  bidict<L, R> result;
  auto try_equate = [&](L const &l, R const &r) {
    if (result.contains_l(l) && result.at_l(l) != r) {
      return false;
    }
    if (result.contains_r(r) && result.at_r(r) != l) {
      return false;
    }
    result.equate(l, r);
    return true;
  };

  for (auto const &[l, r] : d1) {
    if (!try_equate(l, r)) {
      return std::nullopt;
    }
  }

  for (auto const &[l, r] : d2) {
    if (!try_equate(l, r)) {
      return std::nullopt;
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
