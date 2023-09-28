#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_ISOMORPHIC_TO_PAIR_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_ISOMORPHIC_TO_PAIR_H

#include <utility>
#include <tuple>
#include <array>
#include "is_pair.h"

namespace FlexFlow {

template <typename T, typename Enable = void> struct is_isomorphic_to_pair : std::false_type { };
template <typename T> struct is_isomorphic_to_pair<T, std::enable_if_t<is_pair_v<T>>> : std::true_type { };
template <typename T1, typename T2> struct is_isomorphic_to_pair<std::pair<T1, T2>> : std::true_type { };
template <typename T1, typename T2> struct is_isomorphic_to_pair<std::tuple<T1, T2>> : std::true_type { };
template <typename T> struct is_isomorphic_to_pair<std::array<T, 2>> : std::true_type { };

template <typename T>
inline constexpr bool is_isomorphic_to_pair_v = is_isomorphic_to_pair<T>::value;

} // namespace FlexFlow

#endif
