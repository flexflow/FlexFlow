#ifndef _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_FUNCTIONS_TO_TYPE_LIST_H
#define _FLEXFLOW_LIB_UTILS_TYPE_LIST_INCLUDE_UTILS_TYPE_LIST_FUNCTIONS_TO_TYPE_LIST_H

namespace FlexFlow {

template <typename T>
struct to_type_list;

template <typename T>
using to_type_list_t = typename to_type_list<T>::type;

} // namespace FlexFlow

#endif
