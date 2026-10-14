#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_COMPOSE_ONE_TO_MANYS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_COMPOSE_ONE_TO_MANYS_H

#include "utils/one_to_many/exhaustive_relational_join.h"

namespace FlexFlow {

template <typename T1, typename T2, typename T3>
OneToMany<T1, T3> compose_one_to_manys(OneToMany<T1, T2> const &fst,
                                       OneToMany<T2, T3> const &snd) {
  return exhaustive_relational_join(fst, snd);
}

} // namespace FlexFlow

#endif
