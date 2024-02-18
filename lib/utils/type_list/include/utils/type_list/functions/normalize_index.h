#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_NORMALIZE_INDEX_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_NORMALIZE_INDEX_H

#include "utils/type_list/type_list.h"
#include <type_traits>
#include "length.h"

namespace FlexFlow {

template <int Length, int N, typename Enable = void> struct normalize_index { };

template <int Length, int N>
struct normalize_index<Length, N, std::enable_if_t<(N >= 0) && (N < Length)>>
  : std::integral_constant<int, N> { };

template <int Length, int N>
struct normalize_index<Length, N, std::enable_if_t<(N < 0) && ((-N + 1) < Length)>>
  : normalize_index<Length, (Length + N)> { };

template <int Length, int N>
inline constexpr int normalize_index_v = normalize_index<Length, N>::value;

} // namespace FlexFlow

#endif
