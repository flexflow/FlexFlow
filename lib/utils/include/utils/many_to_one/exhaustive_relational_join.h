#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_EXHAUSTIVE_RELATIONAL_JOIN_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_EXHAUSTIVE_RELATIONAL_JOIN_H

#include "utils/exception.h"
#include "utils/many_to_one/many_to_one.h"

namespace FlexFlow {

template <typename T1, typename T2, typename T3>
ManyToOne<T1, T3> exhaustive_relational_join(ManyToOne<T1, T2> const &fst,
                                             ManyToOne<T2, T3> const &snd) {
  ManyToOne<T1, T3> result;

  if (fst.right_values() != snd.left_values()) {
    PANIC(
        fmt::format("exhaustive_relational_join for ManyToOne received inputs "
                    "with non-matching inner dimensions: right dimension of "
                    "fst is {} while left dimension of snd is {}",
                    fst.right_values(),
                    snd.left_values()));
  }

  for (T3 const &t3 : snd.right_values()) {
    for (T2 const &t2 : snd.at_r(t3)) {
      for (T1 const &t1 : fst.at_r(t2)) {
        result.insert({t1, t3});
      }
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
