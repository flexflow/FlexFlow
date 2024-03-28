#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_TYPE_FUNCTIONS_TYPE_AT_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_TYPE_FUNCTIONS_TYPE_AT_H

#include "utils/backports/type_identity.h"
#include "visit_struct/visit_struct.hpp"
#include <cstddef>

namespace FlexFlow {

template <size_t Idx, typename T>
struct type_at : type_identity<::visit_struct::type_at<Idx, T>> {};

template <size_t Idx, typename T>
using type_at_t = typename type_at<Idx, T>::type;

} // namespace FlexFlow

#endif
