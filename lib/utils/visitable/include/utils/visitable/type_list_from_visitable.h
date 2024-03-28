#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_AS_TYPE_LIST_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_AS_TYPE_LIST_H

#include "utils/type_list/type_list.h"
#include "utils/type_list/prepend.h"
#include "utils/visitable/is_visitable.h"
#include "utils/visitable/field_count.h"
#include "utils/visitable/required.h"
#include "utils/type_list/transform.h"

namespace FlexFlow {

template <typename T, int i, typename Enable = void>
struct type_list_from_visitable_raw_helper;

template <typename T, int i>
using type_list_from_visitable_raw_helper_t =
    typename type_list_from_visitable_raw_helper<T, i>::type;

template <typename T, int i>
struct type_list_from_visitable_raw_helper<T, i, std::enable_if_t<(i < visitable_field_count_v<T>)>>
    : type_list_prepend<visit_struct::type_at<i, T>,
                        type_list_from_visitable_raw_helper_t<T, i + 1>> {};

template <typename T, int i>
struct type_list_from_visitable_raw_helper<T,
                                 i,
                                 std::enable_if_t<(i == visitable_field_count_v<T>)>>
    : type_identity<type_list<>> {};

template <typename T>
using type_list_from_visitable_raw = type_list_from_visitable_raw_helper<T, 0>;

template <typename T>
using type_list_from_visitable_raw_t = typename type_list_from_visitable_raw<T>::type;

template <typename T>
using type_list_from_visitable = type_list_transform<remove_req, type_list_from_visitable_raw_t<T>>;

template <typename T>
using type_list_from_visitable_t = typename type_list_from_visitable<T>::type;

} // namespace FlexFlow

#endif
