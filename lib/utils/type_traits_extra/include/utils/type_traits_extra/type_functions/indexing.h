#ifndef _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_HEAD_T_H
#define _FLEXFLOW_LIB_UTILS_TYPE_TRAITS_EXTRA_INCLUDE_UTILS_TYPE_TRAITS_EXTRA_TYPE_LIST_HEAD_T_H

namespace FlexFlow {

// naming conventions following http://s3.amazonaws.com/lyah/listmonster.png

template <typename T>
struct get_head { };

template <typename T>
using get_head_t = typename get_head<T>::type;

template <typename T>
struct get_tail { };

template <typename T>
using get_tail_t = typename get_tail<T>::type;

template <typename T>
struct get_last { };

template <typename T>
using get_last_t = typename get_last<T>::type;

template <typename T>
struct get_init { };

template <typename T>
using get_init_t = typename get_init<T>::type;

} // namespace FlexFlow

#endif
