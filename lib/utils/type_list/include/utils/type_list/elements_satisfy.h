#ifndef _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_FUNCTIONS_ELEMENTS_SATISFY_H
#define _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_FUNCTIONS_ELEMENTS_SATISFY_H

#include <type_traits>
#include "utils/type_list/type_list.h"
#include "utils/type_traits_extra/is_nary_typelevel_function.h"

namespace FlexFlow {

template <template <typename...> class Cond, typename... Ts>
struct elements_satisfy_impl;

template <template <typename...> class Cond, typename Head, typename... Ts>
struct elements_satisfy_impl<Cond, Head, Ts...>
    : std::conjunction<Cond<Head>, elements_satisfy_impl<Cond, Ts...>> {};

template <template <typename...> class Cond>
struct elements_satisfy_impl<Cond> : std::true_type {};

template <template <typename...> class Cond, typename T>
struct elements_satisfy { };

template <template <typename...> class Cond, typename... Ts>
struct elements_satisfy<Cond, type_list<Ts...>>
    : elements_satisfy_impl<Cond, Ts...> 
{
  static_assert(is_nary_typelevel_function_v<Cond, 1>,
                "Cannot call elements_satisfy with a type-level function with more "
                "than 1 argument");
};

template <template <typename...> class cond, typename t>
inline constexpr bool elements_satisfy_v = elements_satisfy<cond, t>::value;


/* static_assert(elements_satisfy_v<is_equal_comparable, std::tuple<int,
 * float>>); */

/* template <template <typename...> class Cond, typename T, typename Enable> */
/* struct violating_element { */
/*   static_assert(false, */
/*                 "Cannot call violating_element with a metafunction with more " */
/*                 "than 1 argument"); */
/* }; */


} // namespace FlexFlow

#endif
