#ifndef _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_TYPECLASS_TYPE_LIST_ISOMORPHISM_FUNCTIONS_TYPE_CAT_H
#define _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_TYPECLASS_TYPE_LIST_ISOMORPHISM_FUNCTIONS_TYPE_CAT_H

#include "utils/type_list/typeclass/type_list_adjunction/definition.h"

namespace FlexFlow {

template <typename T1, typename T2, typename Instance = default_type_list_adjunction_t<T1>, typename Enable = void>
struct type_cat { };

template <typename T1, typename T2, typename Instance>
struct type_cat<T1, T2, Instance, std::enable_if_t<is_valid_type_list_adjunction_v<T, Instance> && is_valid_type_list_adjunction_v<T, Instance>>>
  : from_type_list<Instance, type_list_concat_t<to_type_list_t<T1, Instance>, to_type_list_t<T2, Instance>>> { };

template <typename T1, typename T2, typename Instance = default_type_list_adjunction_t<T1>>
using type_cat_t = typename type_cat<T1, T2, Instance>::type;

} // namespace FlexFlow

#endif
