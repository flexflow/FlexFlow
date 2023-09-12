#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATIONS_FROM_TUPLE_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATIONS_FROM_TUPLE_H

namespace FlexFlow {

template <typename T>
struct use_visitable_constructor {
  template <typename... Args,
            typename = typename std::enable_if<std::is_same<
                std::tuple<Args...>,
                typename visit_struct::type_at<0, T>::type>::value>::type>
  use_visitable_constructor(Args &&...args) {
    visit_construct<T, Args...>(*this, std::forward<Args>(args)...);
  }
};

template <typename T, typename... Args>
void visit_construct_tuple(T &t, visit_as_tuple_t<T> const &tup) {
  static_assert(is_visitable<T>::value, "Type must be visitable");

  construct_visitor<T> vis{t, tup};
  visit_struct::visit_pointers<T>(vis);
}

template <typename T, typename Tup, int... S>
T visitable_from_tuple_impl(seq<S...>, Tup const &tup) {
  return T{std::get<S>(tup)...};
}

template <typename T, typename... Args>
T visitable_from_tuple(std::tuple<Args...> const &t) {
  return visitable_from_tuple_impl<T>(seq_enumerate_args_t<Args...>{}, t);
};


} // namespace FlexFlow

#endif
