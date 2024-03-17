#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_TRANSFORM_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_TRANSFORM_H

#include "utils/backports/type_identity.h"
#include "utils/type_list/type/tuple.h"
#include "utils/type_list/type/variant.h"
#include "utils/type_list/type_list.h"
#include "utils/type_traits_extra/metafunction/is_nary.h"

namespace FlexFlow {

template <template <typename...> class Func, typename T, typename Enable = void>
struct transform {};

template <template <typename...> class Func, typename T>
using transform_t = typename transform<Func, T>::type;

template <template <typename...> class Func, typename... Args>
struct transform<Func,
                 type_list<Args...>,
                 std::enable_if_t<is_nary_metafunction_v<Func, 1>>>
    : type_identity<type_list<typename Func<Args>::type...>> {};

template <template <typename...> class Func, typename... Args>
struct transform<Func,
                 std::tuple<Args...>,
                 std::enable_if_t<is_nary_metafunction_v<Func, 1>>>
    : from_type_list<transform_t<Func, type_list<Args...>>> {};

template <template <typename...> class Func, typename... Args>
struct transform<Func,
                 std::variant<Args...>,
                 std::enable_if_t<is_nary_metafunction_v<Func, 1>>>
    : from_type_list<transform_t<Func, type_list<Args...>>> {};

} // namespace FlexFlow

#endif
