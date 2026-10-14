#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_COMPOSE_BIDICTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_COMPOSE_BIDICTS_H

#include "utils/bidict/algorithms/exhaustive_relational_join.h"

namespace FlexFlow {

template <typename T1, typename T2, typename T3>
bidict<T1, T3> compose_bidicts(bidict<T1, T2> const &fst,
                               bidict<T2, T3> const &snd) {
  return exhaustive_relational_join(fst, snd);
}

} // namespace FlexFlow

#endif
