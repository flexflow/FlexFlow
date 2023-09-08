#ifndef _FLEXFLOW_LIB_UTILS_STRONG_TYPEDEF_INCLUDE_UTILS_STRONG_TYPEDEF_UNDERLYING_TYPE_H
#define _FLEXFLOW_LIB_UTILS_STRONG_TYPEDEF_INCLUDE_UTILS_STRONG_TYPEDEF_UNDERLYING_TYPE_H

#include "strong_typedef.h"
#include "utils/backports/type_identity.h" 

namespace FlexFlow {

template <typename Tag, typename T>
T underlying_type_impl(strong_typedef<Tag, T>);

template <typename T>
struct underlying_type
    : type_identity<decltype(underlying_type_impl(std::declval<T>()))> {};

template <typename T>
using underlying_type_t = typename underlying_type<T>::type;

} // namespace FlexFlow

#endif
