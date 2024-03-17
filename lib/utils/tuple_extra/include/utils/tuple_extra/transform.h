#ifndef _FLEXFLOW_LIB_UTILS_TUPLE_EXTRA_INCLUDE_UTILS_TUPLE_EXTRA_TRANSFORM_H
#define _FLEXFLOW_LIB_UTILS_TUPLE_EXTRA_INCLUDE_UTILS_TUPLE_EXTRA_TRANSFORM_H

#include <tuple>
#include "utils/compile_time_sequence/enumerate_args.h"
#include "utils/tuple_extra/prepend.h"
#include "utils/type_list/functions/tuple_from_type_list.h"
#include "utils/compile_time_sequence/sequence.h"
#include "utils/compile_time_sequence/transform_type.h"

namespace FlexFlow {

template <typename F, int X, int... S>
auto seq_transform(F const &f, seq<X, S...> const &)
    -> tuple_from_type_list_t<seq_transform_type_t<F, seq<X, S...>>> {
  auto head = f(std::integral_constant<int, X>{});
  auto tail = seq_transform(f, seq<S...>{});
  return tuple_prepend(head, tail);
}

template <typename F>
std::tuple<> seq_transform(F const &f, seq<> const &) {
  return {};
}

template <typename F, typename... Ts>
auto transform(std::tuple<Ts...> const &tup, F const &f) {
  return seq_transform(
      [&](auto idx) { return f(std::get<decltype(idx)::value>(tup)); },
      seq_enumerate_args_t<Ts...>{});
}

} // namespace FlexFlow

#endif
