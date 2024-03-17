#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_IS_VISITABLE_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_IS_VISITABLE_H

#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

template <typename T>
using is_visitable = ::visit_struct::traits::is_visitable<T>;

template <typename T>
inline constexpr bool is_visitable_v = is_visitable<T>::value;

#define CHECK_VISITABLE(...)                                                   \
  static_assert(::FlexFlow::is_visitable_v<__VA_ARGS__>,                       \
                #__VA_ARGS__ " should be visitable (but is not)");             \
  DEBUG_PRINT_TYPE(__VA_ARGS__)

} // namespace FlexFlow

#endif
