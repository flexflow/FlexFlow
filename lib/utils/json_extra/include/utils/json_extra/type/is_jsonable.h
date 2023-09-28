#ifndef _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_TYPE_IS_JSONABLE_H
#define _FLEXFLOW_LIB_UTILS_JSON_EXTRA_INCLUDE_UTILS_JSON_EXTRA_TYPE_IS_JSONABLE_H

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_jsonable
    : conjunction<is_json_serializable<T>, is_json_deserializable<T>> {};

template <typename T>
inline constexpr bool is_jsonable_v = is_jsonable<T>::value;

#define CHECK_IS_JSONABLE(TYPENAME)                                            \
  static_assert(is_json_serializable<TYPENAME>::value,                         \
                #TYPENAME " should be json serializeable");                    \
  static_assert(is_json_deserializable<TYPENAME>::value,                       \
                #TYPENAME " should be json deserializeable")

} // namespace FlexFlow

#endif
