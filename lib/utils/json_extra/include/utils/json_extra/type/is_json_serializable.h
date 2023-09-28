#ifndef _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_TYPE_IS_JSON_SERIALIZABLE_H
#define _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_TYPE_IS_JSON_SERIALIZABLE_H

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_json_serializable : std::false_type {};

template <typename T>
struct is_json_serializable<
    T,
    std::void_t<decltype(std::declval<json>() = std::declval<T>())>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_json_serializable_v = is_json_serializable<T>::value;

} // namespace FlexFlow

#endif
