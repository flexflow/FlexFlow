#ifndef _FLEXFLOW_LIB_UTILS_STRONG_TYPEDEF_INCLUDE_UTILS_STRONG_TYPEDEF_HASH_H
#define _FLEXFLOW_LIB_UTILS_STRONG_TYPEDEF_INCLUDE_UTILS_STRONG_TYPEDEF_HASH_H

#include "utils/type_traits_extra/is_hashable.h"
#include "underlying_type.h"
#include <functional>

namespace FlexFlow {

// derived from
// https://github.com/foonathan/type_safe/blob/3612e2828b4b4e0d1cc689373e63a6d59d4bfd79/include/type_safe/strong_typedef.hpp
template <typename StrongTypedef>
struct hashable : std::hash<underlying_type_t<StrongTypedef>> {
  using underlying_ty = underlying_type_t<StrongTypedef>;
  using underlying_hash = std::hash<underlying_ty>;

  static_assert(is_hashable_v<underlying_ty>);

  std::size_t operator()(StrongTypedef const &lhs) const
      noexcept(noexcept(underlying_hash{}(std::declval<underlying_ty>()))) {
    return underlying_hash{}(static_cast<underlying_ty const &>(lhs));
  }
};

#define MAKE_TYPEDEF_HASHABLE(TYPEDEF_NAME)                                    \
  namespace std {                                                              \
  template <>                                                                  \
  struct hash<TYPEDEF_NAME> : ::FlexFlow::hashable<TYPEDEF_NAME> {};           \
  }                                                                            \
  static_assert(true, "")

#define FF_TYPEDEF_HASHABLE(TYPEDEF_NAME)                                      \
  }                                                                            \
  MAKE_TYPEDEF_HASHABLE(::FlexFlow::TYPEDEF_NAME);                             \
  namespace FlexFlow {                                                         \
  static_assert(true, "");

} // namespace FlexFlow

#endif
