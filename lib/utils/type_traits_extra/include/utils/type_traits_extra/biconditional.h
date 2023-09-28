#ifndef _FLEXFLOW_UTILS_TYPE_TRAITS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_BICONDITIONAL_H
#define _FLEXFLOW_UTILS_TYPE_TRAITS_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_BICONDITIONAL_H

#include <type_traits> 

namespace FlexFlow {

template <typename L, typename R>
struct biconditional : std::bool_constant<(bool(L::value) == bool(R::value))> {};

}

#endif
