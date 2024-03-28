#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATIONS_CONSTRUCT_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATIONS_CONSTRUCT_H

#include "utils/visitable/visit_as_tuple.h"
#include "utils/visitable/is_visitable.h"
#include "utils/visitable/check_visitable.h"
#include <any>
#include <tuple>

namespace FlexFlow {

template <typename C, typename... Args>
struct construct_visitor {
  construct_visitor(C &c, std::tuple<Args const &...> args)
      : c(c), args(args) {}

  std::size_t idx = 0;
  std::tuple<Args const &...> args;
  C &c;

  template <typename T>
  void operator()(char const *, T C::*ptr_to_member) {
    c.*ptr_to_member = std::any_cast<T>(get(args, idx));
    this->idx++;
  };
};

template <typename T, typename... Args>
void visit_construct(T &t, Args &&...args) {
  CHECK_VISITABLE(T);
  static_assert(std::is_same<std::tuple<Args...>, visit_as_tuple_t<T>>::value,
                "");

  std::tuple<Args...> tup(std::forward<Args>(args)...);
  construct_visitor<T> vis{t, tup};
  visit_struct::visit_pointers<T>(vis);
}

template <typename T, typename... Args>
T construct_visitable(Args &&...args) {
  T t(std::forward<Args>(args)...);
  return t;
}

} // namespace FlexFlow

#endif
