#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_FUNCTIONS_PREPEND_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_FUNCTIONS_PREPEND_H

#include <tuple>
#include "utils/backports/type_identity.h"
#include <variant>

namespace FlexFlow {

template <typename Element, typename List>
struct prepend { };

template <typename Element, typename List>
using prepend_t = typename prepend<Element, List>::type;


template <typename Element, typename... Args>
struct prepend<Element, std::tuple<Args...>> : type_identity<std::tuple<Element, Args...>> { };

template <typename Element, typename... Args>
struct prepend<Element, std::variant<Args...>> : type_identity<std::variant<Element, Args...>> { };


} // namespace FlexFlow

#endif
