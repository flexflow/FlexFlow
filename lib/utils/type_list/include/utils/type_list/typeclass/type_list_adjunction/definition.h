#ifndef _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_IS_ISOMORPHIC_TO_TYPE_LIST_H
#define _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_IS_ISOMORPHIC_TO_TYPE_LIST_H

#include "utils/type_list/is_type_list.h" 
#include <type_traits>

namespace FlexFlow {

template <typename T> struct default_type_list_adjunction { };

template <typename T>
using default_type_list_adjunction_t = typename default_type_list_adjunction<T>::type;

template <typename T, typename Instance>
struct is_valid_type_list_adjunction : std::bool_constant<
    is_type_list_v<typename Instance::template to_type_list<T>::type> && 
    std::is_same_v<T, typename Instance::template from_type_list<typename Instance::template to_type_list<T>::type>::type>
> { };

template <typename T, typename Instance>
inline constexpr bool is_valid_type_list_adjunction_v = is_valid_type_list_adjunction<T, Instance>::value;

template <typename T, typename Instance = default_type_list_adjunction_t<T>>
struct to_type_list 
  : Instance::template to_type_list<T> { };

template <typename T, typename Instance = default_type_list_adjunction_t<T>>
using to_type_list_t = typename to_type_list<T, Instance>::type;

template <typename Instance, typename TypeList>
struct from_type_list
  : Instance::template from_type_list<TypeList, Instance> { };

template <typename Instance, typename TypeList>
using from_type_list_t = typename from_type_list<TypeList, Instance>::type;

} // namespace FlexFlow

#endif
