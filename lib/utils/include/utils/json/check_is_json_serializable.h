#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_CHECK_IS_JSON_SERIALIZABLE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JSON_CHECK_IS_JSON_SERIALIZABLE_H

#include "utils/json/is_json_serializable.h"

namespace FlexFlow {

#define CHECK_IS_JSON_SERIALIZABLE(TYPENAME)                                   \
  static_assert(::FlexFlow::is_json_serializable<TYPENAME>::value,             \
                #TYPENAME " should be json serializeable")

} // namespace FlexFlow

#endif
