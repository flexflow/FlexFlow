#ifndef _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_TYPECLASS_TYPE_LIST_ISOMORPHISM_FUNCTIONS_TYPE_SUBSETEQ_H
#define _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_TYPECLASS_TYPE_LIST_ISOMORPHISM_FUNCTIONS_TYPE_SUBSETEQ_H

#include "utils/type_list/typeclass/type_list_adjunction/definition.h"

namespace FlexFlow {

template <typename L, typename R, typename Instance = default_type_list_adjunction_t<L>, typename Enable = void>
struct type_subseteq { };

template <typename L, typename R, typename Instance>
struct type_subseteq<L, R, 
  std::enable_if_t<is_valid_type_list_adjunction_v<L, Instance> && is_valid_type_list_adjunction_v<R, Instance>>
  : type_list_subseteq<to_type_list_t<L, Instance>, to_type_list_t<R, Instance>> { };

template <typename L, typename R, typename Instance = default_type_list_adjunction_t<L>>
inline constexpr bool type_subseteq_v = type_subseteq<L, R, Instance>::value;


} // namespace FlexFlow

#endif
