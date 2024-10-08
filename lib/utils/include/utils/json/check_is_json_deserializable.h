#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_CHECK_IS_JSON_DESERIALIZABLE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_CHECK_IS_JSON_DESERIALIZABLE_H

#include "utils/json/is_json_deserializable.h"

namespace FlexFlow {

#define CHECK_IS_JSON_DESERIALIZABLE(TYPENAME)                                 \
  static_assert(::FlexFlow::is_json_deserializable<TYPENAME>::value,           \
                #TYPENAME " should be json deserializeable")

} // namespace FlexFlow

#endif
