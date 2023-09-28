#ifndef _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_LOOKUP_H
#define _FLEXFLOW_LIB_UTILS_BIDICT_INCLUDE_UTILS_BIDICT_ALGORITHMS_LOOKUP_H

namespace FlexFlow {

template <typename L, typename R>
std::function<R(L const &)> lookup_in_l(bidict<L, R> const &m) {
  return [&m](L const &l) -> R { return m.at_l(l); };
}

template <typename L, typename R>
std::function<L(R const &)> lookup_in_r(bidict<L, R> const &m) {
  return [&m](R const &r) -> L { return m.at_r(r); };
}


} // namespace FlexFlow

#endif
