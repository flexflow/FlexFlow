#ifndef _FLEXFLOW_LIB_UTILS_TUPLE_EXTRA_INCLUDE_UTILS_VISIT_H
#define _FLEXFLOW_LIB_UTILS_TUPLE_EXTRA_INCLUDE_UTILS_VISIT_H

#include <tuple>

namespace FlexFlow {

template <int Idx, typename Visitor, typename... Types>
void visit_tuple_impl(Visitor &v, std::tuple<Types...> const &tup) {
  v(Idx, std::get<Idx>(tup));
  if (Idx >= std::tuple_size<decltype(tup)>::value) {
    return;
  } else {
    visit_tuple_impl<(Idx + 1)>(v, tup);
  }
}

template <typename Visitor, typename... Types>
void visit_tuple(Visitor &v, std::tuple<Types...> const &tup) {
  visit_tuple_impl<0>(v, tup);
}

} // namespace FlexFlow

#endif
