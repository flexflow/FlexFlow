#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_SEQUENCE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_SEQUENCE_H

#include <utility>

namespace FlexFlow {

template <int ...S>
struct seq { };

template <int X, int ...S>
struct seq_prepend {
  using type = seq<X, S...>;
};

template <typename Rest, int Head> struct seq_append;

template <int X, int ...S>
struct seq_append<seq<S...>, X>  {
  using type = seq<S..., X>;
};

template <int n>
struct seq_count {
  using type = typename seq_append<typename seq_count<(n-1)>::type, n>::type;
};

template <>
struct seq_count<0> {
  using type = seq<>;
};

template <int n>
using seq_count_t = typename seq_count<n>::type;

template <typename F, int X, int ...S>
auto seq_get(F const &f, int i, seq<X, S...> const &s) -> decltype(f(std::declval<std::integral_constant<int, 0>>())) {
  if (X == i) {
    return f(std::integral_constant<int, X>{});
  } else {
    return seq_get(f, i+1, s);
  }
};

template <typename F>
auto seq_get(F const &f, int i, seq<> const &) -> decltype(f(std::declval<std::integral_constant<int, 0>>())) {
  throw mk_runtime_error("Failed seq_get for index {}", i);
}

}

#endif
