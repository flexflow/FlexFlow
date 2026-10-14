#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_EXHAUSTIVE_RELATIONAL_JOIN_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_EXHAUSTIVE_RELATIONAL_JOIN_H

#include "utils/exception.h"
#include "utils/one_to_many/one_to_many.h"

namespace FlexFlow {

template <typename T1, typename T2, typename T3>
OneToMany<T1, T3> exhaustive_relational_join(OneToMany<T1, T2> const &fst,
                                             OneToMany<T2, T3> const &snd) {
  OneToMany<T1, T3> result;

  if (fst.right_values() != snd.left_values()) {
    throw mk_runtime_error(
        fmt::format("exhaustive_relational_join for OneToMany received inputs "
                    "with non-matching inner dimensions: right dimension of "
                    "fst is {} while left dimension of snd is {}",
                    fst.right_values(),
                    snd.left_values()));
  }

  for (T1 const &t1 : fst.left_values()) {
    for (T2 const &t2 : fst.at_l(t1)) {
      for (T3 const &t3 : snd.at_l(t2)) {
        result.insert({t1, t3});
      }
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
