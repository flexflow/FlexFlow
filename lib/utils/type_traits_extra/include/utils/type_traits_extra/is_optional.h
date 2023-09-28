#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_OPTIONAL_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_IS_OPTIONAL_H

#include <optional>
#include "is_decayable.h" 

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_optional : std::false_type { };

template <typename T>
struct is_optional<std::optional<T>> : std::true_type { };

template <typename T>
struct is_optional<T, std::enable_if_t<is_decayable_v<T>>> : is_optional<std::decay_t<T>> { };

template <>
struct is_optional<std::nullopt_t> : std::true_type { };

template <typename T>
inline constexpr bool is_optional_v = is_optional<T>::value;

} // namespace FlexFlow

#endif
