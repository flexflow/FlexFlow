#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_PAIR_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_PAIR_H

#include <array>
#include <tuple>
#include <utility>

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_pair : std::false_type {};
template <typename T1, typename T2>
struct is_pair<std::pair<T1, T2>> : std::true_type {};
template <typename T>
inline constexpr bool is_pair_v = is_pair<T>::value;

} // namespace FlexFlow

#endif
