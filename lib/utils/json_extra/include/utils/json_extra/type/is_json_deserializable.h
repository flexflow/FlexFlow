#ifndef _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_TYPE_IS_JSON_DESERIALIZABLE_H
#define _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_TYPE_IS_JSON_DESERIALIZABLE_H

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_json_deserializable : std::false_type {};

template <typename T>
struct is_json_deserializable<T,
                              std::void_t<decltype(std::declval<json>().get<T>())>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_json_deserializable_v = is_json_deserializable<T>::value;


} // namespace FlexFlow

#endif
