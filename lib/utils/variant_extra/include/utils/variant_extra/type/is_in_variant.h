#ifndef _FLEXFLOW_LIB_UTILS_VARIANT_EXTRA_INCLUDE_UTILS_VARIANT_EXTRA_TYPE_IS_IN_VARIANT_H
#define _FLEXFLOW_LIB_UTILS_VARIANT_EXTRA_INCLUDE_UTILS_VARIANT_EXTRA_TYPE_IS_IN_VARIANT_H

#include "utils/type_traits_extra/type_list/as_type_list.h"
#include "utils/type_traits_extra/type_list/contains.h"

namespace FlexFlow {

template <typename T, typename Variant>
struct is_in_variant : type_list_contains<T, as_type_list_t<Variant>> {};

template <typename T, typename Variant>
inline constexpr bool is_in_variant_v = is_in_variant<T, Variant>::value;

} // namespace FlexFlow

#endif
