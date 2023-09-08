#ifndef _FLEXFLOW_LIB_UTILS_STRONG_TYPEDEF_INCLUDE_UTILS_STRONG_TYPEDEF_HASH_H
#define _FLEXFLOW_LIB_UTILS_STRONG_TYPEDEF_INCLUDE_UTILS_STRONG_TYPEDEF_HASH_H

#include "underlying_type.h"
#include <functional>

namespace FlexFlow {

// derived from
// https://github.com/foonathan/type_safe/blob/3612e2828b4b4e0d1cc689373e63a6d59d4bfd79/include/type_safe/strong_typedef.hpp
template <typename StrongTypedef>
struct hashable : std::hash<underlying_type_t<StrongTypedef>> {
  using underlying_ty = underlying_type_t<StrongTypedef>;
  using underlying_hash = std::hash<underlying_ty>;

  std::size_t operator()(StrongTypedef const &lhs) const
      noexcept(noexcept(underlying_hash{}(std::declval<underlying_ty>()))) {
    return underlying_hash{}(static_cast<underlying_ty const &>(lhs));
  }
};


} // namespace FlexFlow

#endif
