#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_IS_JSON_SERIALIZABLE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_IS_JSON_SERIALIZABLE_H

#include "utils/type_traits_core.h"
#include <nlohmann/json.hpp>
#include <type_traits>

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_json_serializable : std::false_type {};

template <typename T>
struct is_json_serializable<
    T,
    void_t<decltype(std::declval<nlohmann::json>() = std::declval<T>())>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_json_serializable_v = is_json_serializable<T>::value;

} // namespace FlexFlow

#endif
