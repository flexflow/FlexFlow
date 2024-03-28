#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATIONS_AS_TUPLE_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATIONS_AS_TUPLE_H

#include "utils/visitable/visit_as_tuple.h"

namespace FlexFlow {

template <typename T>
struct GetFunctor {
  GetFunctor(T const &t) : t(t) {}

  T const &t;

  template <int IDX>
  auto operator()(std::integral_constant<int, IDX> const &) const
      -> remove_req_t<decltype(visit_struct::get<IDX>(t))> {
    return visit_struct::get<IDX>(t);
  }
};

template <typename T>
visit_as_tuple_t<T> tuple_from_visitable(T const &t) {
  GetFunctor<T> func(t);
  return seq_transform(func, seq_enumerate_tuple_t<visit_as_tuple_t<T>>{});
}

} // namespace FlexFlow

#endif
