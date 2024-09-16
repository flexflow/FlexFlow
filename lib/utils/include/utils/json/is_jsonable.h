#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_IS_JSONABLE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_IS_JSONABLE_H

#include "utils/json/is_json_deserializable.h"
#include "utils/json/is_json_serializable.h"

namespace FlexFlow {

template <typename T, typename Enable = void>
struct is_jsonable
    : std::conjunction<is_json_serializable<T>, is_json_deserializable<T>> {};

template <typename T>
inline constexpr bool is_jsonable_v = is_jsonable<T>::value;

} // namespace FlexFlow

#endif
