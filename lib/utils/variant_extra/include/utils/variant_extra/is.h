#ifndef _FLEXFLOW_LIB_UTILS_VARIANT_EXTRA_INCLUDE_UTILS_VARIANT_EXTRA_IS_H
#define _FLEXFLOW_LIB_UTILS_VARIANT_EXTRA_INCLUDE_UTILS_VARIANT_EXTRA_IS_H

#include "type/is_in_variant.h"
#include <variant>

namespace FlexFlow {

template <typename T, typename... TRest, typename... Args>
bool is(std::variant<Args...> const &v) {
  static_assert(is_in_variant<T, std::variant<Args...>>::value);

  return std::holds_alternative<T>(v) || is<TRest...>(v);
}

} // namespace FlexFlow

#endif
