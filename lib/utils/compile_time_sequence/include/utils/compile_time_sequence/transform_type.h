#ifndef _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_TRANSFORM_TYPE_H
#define _FLEXFLOW_LIB_UTILS_COMPILE_TIME_SEQUENCE_INCLUDE_UTILS_COMPILE_TIME_SEQUENCE_TRANSFORM_TYPE_H

#include "utils/compile_time_sequence/sequence.h"
#include "utils/type_list/prepend.h"
#include "utils/type_list/type_list.h"
#include <type_traits>

namespace FlexFlow {

template <typename F, typename Seq>
struct seq_transform_type;

template <typename F, int X, int... S>
struct seq_transform_type<F, seq<X, S...>>
    : type_list_prepend<
          std::remove_reference_t<std::remove_cv_t<decltype(std::declval<F>()(
              std::declval<std::integral_constant<int, X>>()))>>,
          typename seq_transform_type<F, seq<S...>>::type> {};

template <typename F>
struct seq_transform_type<F, seq<>> {
  using type = type_list<>;
};

template <typename F, typename Seq>
using seq_transform_type_t = typename seq_transform_type<F, Seq>::type;

} // namespace FlexFlow

#endif
