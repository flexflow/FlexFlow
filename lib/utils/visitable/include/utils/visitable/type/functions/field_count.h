#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_TYPE_FUNCTIONS_FIELD_COUNT_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_TYPE_FUNCTIONS_FIELD_COUNT_H

#include "visit_struct/visit_struct.hpp"
#include <cstddef>

namespace FlexFlow {

template <typename T>
struct field_count : std::integral_constant<
                         size_t,
                         ::visit_struct::traits::visitable<T>::field_count> {};

template <typename T>
inline constexpr size_t field_count_v = field_count<T>::value;

} // namespace FlexFlow

#endif
