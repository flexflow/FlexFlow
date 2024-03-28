#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_APPLY_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_APPLY_H

#include "utils/type_list/type_list.h"
#include "utils/backports/type_identity.h"
#include "utils/type_traits_extra/is_nary_typelevel_function.h"

namespace FlexFlow {

template <template <typename...> class Func, typename T>
struct type_list_transform {};

template <template <typename...> class Func, typename T>
using type_list_transform_t = typename type_list_transform<Func, T>::type;

template <template <typename...> class Func, typename... Args>
struct type_list_transform<Func, type_list<Args...>>
    : type_identity<type_list<typename Func<Args>::type...>> {
  static_assert(is_nary_typelevel_function_v<Func, 1>);
};

} // namespace FlexFlow

#endif
