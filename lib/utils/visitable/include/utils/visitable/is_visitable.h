#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_IS_VISITABLE_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_VISITABLE_IS_VISITABLE_H

#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

template <typename T>
using is_visitable = ::visit_struct::traits::is_visitable<T>;

template <typename T>
inline constexpr bool is_visitable_v = is_visitable<T>::value;

} // namespace FlexFlow

#endif
