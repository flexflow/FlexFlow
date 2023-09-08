#ifndef _FLEXFLOW_UTILS_VISITABLE_INCLUDE_VISITABLE_AS_TUPLE_H
#define _FLEXFLOW_UTILS_VISITABLE_INCLUDE_VISITABLE_AS_TUPLE_H

#include <type_traits>
#include "visit_struct/visit_struct.hpp"
#include "required.h"
#include "utils/tuple.h"

namespace FlexFlow {

template <typename T>
struct field_count : std::integral_constant<
                         std::size_t,
                         ::visit_struct::traits::visitable<T>::field_count> {};


template <typename T, int i, typename Enable = void>
struct visit_as_tuple_helper;

template <typename T, int i>
struct visit_as_tuple_helper<
    T,
    i,
    typename std::enable_if<(
        i < visit_struct::traits::visitable<T>::field_count)>::type> {
  using type = typename tuple_prepend_type<
      remove_req_t<visit_struct::type_at<i, T>>,
      typename visit_as_tuple_helper<T, i + 1>::type>::type;
};

template <typename T, int i>
struct visit_as_tuple_helper<
    T,
    i,
    typename std::enable_if<(
        i == visit_struct::traits::visitable<T>::field_count)>::type> {
  using type = std::tuple<>;
};

template <typename T, int i, typename Enable = void>
struct visit_as_tuple_raw_helper;

template <typename T, int i>
struct visit_as_tuple_raw_helper<
    T,
    i,
    typename std::enable_if<(
        i < visit_struct::traits::visitable<T>::field_count)>::type> {
  using type = typename tuple_prepend_type<
      visit_struct::type_at<i, T>,
      typename visit_as_tuple_raw_helper<T, i + 1>::type>::type;
};

template <typename T, int i>
struct visit_as_tuple_raw_helper<
    T,
    i,
    typename std::enable_if<(
        i == visit_struct::traits::visitable<T>::field_count)>::type> {
  using type = std::tuple<>;
};

template <typename T>
using visit_as_tuple = visit_as_tuple_helper<T, 0>;

template <typename T>
using visit_as_tuple_raw = visit_as_tuple_raw_helper<T, 0>;

template <typename T>
using visit_as_tuple_t = typename visit_as_tuple<T>::type;

template <typename T>
using visit_as_tuple_raw_t = typename visit_as_tuple_raw<T>::type;

}

#endif
