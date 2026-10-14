#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_COMPOSE_MANY_TO_ONES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_COMPOSE_MANY_TO_ONES_H

#include "utils/many_to_one/exhaustive_relational_join.h"

namespace FlexFlow {

template <typename T1, typename T2, typename T3>
ManyToOne<T1, T3> compose_many_to_ones(ManyToOne<T1, T2> const &fst,
                                       ManyToOne<T2, T3> const &snd) {
  return exhaustive_relational_join(fst, snd);
}

} // namespace FlexFlow

#endif
