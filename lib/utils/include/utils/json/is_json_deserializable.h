#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_IS_JSON_DESERIALIZABLE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_IS_JSON_DESERIALIZABLE_H

#include <type_traits>
#include <nlohmann/json.hpp>
#include "utils/type_traits_core.h"

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_json_deserializable : std::false_type {};

template <typename T>
struct is_json_deserializable<T,
                              void_t<decltype(std::declval<nlohmann::json>().get<T>())>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_json_deserializable_v = is_json_deserializable<T>::value;

} // namespace FlexFlow

#endif
