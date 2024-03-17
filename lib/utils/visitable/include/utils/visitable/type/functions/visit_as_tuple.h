#ifndef _FLEXFLOW_UTILS_VISITABLE_INCLUDE_VISITABLE_AS_TUPLE_H
#define _FLEXFLOW_UTILS_VISITABLE_INCLUDE_VISITABLE_AS_TUPLE_H

#include "field_count.h"
#include "utils/type_traits_extra/type_functions/prepend.h"
#include "utils/type_traits_extra/type_functions/transform.h"
#include "utils/visitable/required.h"
#include "visit_struct/visit_struct.hpp"
#include <type_traits>

namespace FlexFlow {

template <typename T, int i, typename Enable = void>
struct visit_as_tuple_raw_helper;

template <typename T, int i>
using visit_as_tuple_raw_helper_t =
    typename visit_as_tuple_raw_helper<T, i>::type;

template <typename T, int i>
struct visit_as_tuple_raw_helper<T, i, std::enable_if_t<(i < field_count_v<T>)>>
    : prepend<visit_struct::type_at<i, T>,
              visit_as_tuple_raw_helper_t<T, i + 1>> {};

template <typename T, int i>
struct visit_as_tuple_raw_helper<T,
                                 i,
                                 std::enable_if_t<(i == field_count_v<T>)>>
    : type_identity<std::tuple<>> {};

template <typename T>
using visit_as_tuple_raw = visit_as_tuple_raw_helper<T, 0>;

template <typename T>
using visit_as_tuple_raw_t = typename visit_as_tuple_raw<T>::type;

template <typename T>
using visit_as_tuple = transform<remove_req, visit_as_tuple_raw_t<T>>;

template <typename T>
using visit_as_tuple_t = typename visit_as_tuple<T>::type;

} // namespace FlexFlow

#endif
