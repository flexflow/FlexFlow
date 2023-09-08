#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_CORE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_TYPE_TRAITS_CORE_H

#include <iterator>
#include <type_traits>

namespace FlexFlow {

template <int idx>
struct infinite_recursion {
  using type = typename infinite_recursion<(idx + 1)>::type;
};

template <typename... Args>
/* struct pack {}; */

/* template <typename T, typename... Args> */
/* struct pack_contains_type; */

/* template <typename T, typename Head, typename... Rest> */
/* struct pack_contains_type<T, Head, Rest...> */
/*     : disjunction<std::is_same<T, Head>, pack_contains_type<T, Rest...>> {}; */

/* template <typename T, typename... Args> */
/* struct pack_contains_type<T, pack<Args...>> : pack_contains_type<T, Args...> {}; */

/* template <typename T> */
/* struct pack_contains_type<T> : std::false_type {}; */

static_assert(pack_contains_type<int, float, double, int, char>::value, "");
static_assert(!pack_contains_type<int, float, double, char>::value, "");

} // namespace FlexFlow

#endif
