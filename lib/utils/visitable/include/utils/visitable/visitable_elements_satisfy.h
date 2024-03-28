#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_VISITABLE_ELEMENTS_SATISFY_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_VISITABLE_ELEMENTS_SATISFY_H

#include "utils/type_list/elements_satisfy.h"
#include "utils/visitable/type_list_from_visitable.h"

namespace FlexFlow {

template <template <typename...> class Cond, typename T>
struct visitable_elements_satisfy
  : elements_satisfy<Cond, type_list_from_visitable_t<T>>
{ };

template <template <typename...> class Cond, typename T>
inline constexpr bool visitable_elements_satisfy_v = visitable_elements_satisfy<Cond, T>::value;

} // namespace FlexFlow

#endif
