#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_SEQUENCE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_SEQUENCE_H

#include "utils/tuple.h"
#include "utils/visitable_core.h"
#include <optional>
#include <utility>

namespace FlexFlow {

template <int... S>
struct seq {};

template <typename Seq>
struct seq_head;

template <int X, int... S>
struct seq_head<seq<X, S...>> : std::integral_constant<int, X> {};

template <>
struct seq_head<seq<>> : std::integral_constant<int, -1> {};

template <typename Seq>
struct seq_tail;

template <int X, int... S>
struct seq_tail<seq<X, S...>> {
  using type = seq<S...>;
};

template <>
struct seq_tail<seq<>> {
  using type = seq<>;
};

template <int X, int... S>
struct seq_prepend {
  using type = seq<X, S...>;
};

template <typename Rest, int Head>
struct seq_append;

template <int X, int... S>
struct seq_append<seq<S...>, X> {
  using type = seq<S..., X>;
};

template <int n>
struct seq_count {
  using type = typename seq_append<typename seq_count<(n - 1)>::type, n>::type;
};

template <>
struct seq_count<-1> {
  using type = seq<>;
};

template <int n>
using seq_count_t = typename seq_count<n>::type;

template <typename... Args>
struct seq_enumerate_args {
  using type = seq_count_t<(int)(sizeof...(Args)) - 1>;
};

template <typename... Args>
using seq_enumerate_args_t = typename seq_enumerate_args<Args...>::type;

template <typename T>
struct seq_enumerate;

template <typename... Args>
struct seq_enumerate<std::tuple<Args...>> : seq_enumerate_args<Args...> {};

template <typename T>
using seq_enumerate_t = typename seq_enumerate<T>::type;

template <typename F, typename Seq>
struct seq_transform_type;

template <typename F, int X, int... S>
struct seq_transform_type<F, seq<X, S...>>
    : tuple_prepend_type<
          visit_struct::traits::clean_t<decltype(std::declval<F>()(
              std::declval<std::integral_constant<int, X>>()))>,
          typename seq_transform_type<F, seq<S...>>::type> {};

template <typename F>
struct seq_transform_type<F, seq<>> {
  using type = std::tuple<>;
};

template <typename F, typename Seq>
using seq_transform_type_t = typename seq_transform_type<F, Seq>::type;

template <typename F, int X, int... S>
auto seq_transform(F const &f, seq<X, S...> const &)
    -> seq_transform_type_t<F, seq<X, S...>> {
  return tuple_prepend(f(std::integral_constant<int, X>{}),
                       seq_transform(f, seq<S...>{}));
}

template <typename F>
std::tuple<> seq_transform(F const &f, seq<> const &) {
  return {};
}

template <typename F, int X, int... S>
auto seq_select(F const &f, int i, seq<X, S...> const &s)
    -> decltype(f(std::declval<std::integral_constant<int, 0>>())) {
  if (i == 0) {
    return f(std::integral_constant<int, X>{});
  } else {
    return seq_select(f, i - 1, seq<S...>{});
  }
}

template <typename F>
auto seq_select(F const &f, int i, seq<> const &)
    -> decltype(f(std::declval<std::integral_constant<int, 0>>())) {
  return std::nullopt;
}

template <typename F, int X, int... S>
auto seq_get(F const &f, int i, seq<X, S...> const &s)
    -> decltype(f(std::declval<std::integral_constant<int, 0>>())) {
  if (X == i) {
    return f(std::integral_constant<int, X>{});
  } else {
    return seq_get(f, i, seq<S...>{});
  }
};

template <typename F>
auto seq_get(F const &f, int i, seq<> const &)
    -> decltype(f(std::declval<std::integral_constant<int, 0>>())) {
  throw mk_runtime_error(fmt::format("Failed seq_get for index {}", i));
}

} // namespace FlexFlow

#endif
