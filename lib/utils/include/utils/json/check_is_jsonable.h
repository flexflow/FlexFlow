#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_CHECK_IS_JSONABLE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_CHECK_IS_JSONABLE_H

#include "utils/json/is_json_deserializable.h"
#include "utils/json/is_json_serializable.h"

namespace FlexFlow {

#define CHECK_IS_JSONABLE(TYPENAME)                                            \
  static_assert(is_json_serializable<TYPENAME>::value,                         \
                #TYPENAME " should be json serializeable");                    \
  static_assert(is_json_deserializable<TYPENAME>::value,                       \
                #TYPENAME " should be json deserializeable")

} // namespace FlexFlow

#endif
