#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_DECAYABLE_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_DECAYABLE_H

#include <type_traits>

namespace FlexFlow {

template <typename T>
struct is_decayable : std::negation<std::is_same<std::decay_t<T>, T>> {};

template <typename T>
inline constexpr bool is_decayable_v = is_decayable<T>::value;

}

#endif
