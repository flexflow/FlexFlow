#ifndef _FLEXFLOW_LIB_UTILS_VARIANT_EXTRA_INCLUDE_UTILS_VARIANT_EXTRA_TYPE_IS_VARIANT_H
#define _FLEXFLOW_LIB_UTILS_VARIANT_EXTRA_INCLUDE_UTILS_VARIANT_EXTRA_TYPE_IS_VARIANT_H

#include <variant>

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_variant_impl : std::false_type {};

template <typename... Ts>
struct is_variant_impl<std::variant<Ts...>> : std::true_type {};

template <typename T>
struct is_variant : is_variant_impl<std::decay_t<T>> {};

template <typename T>
inline constexpr bool is_variant_v = is_variant<T>::value;

} // namespace FlexFlow

#endif
