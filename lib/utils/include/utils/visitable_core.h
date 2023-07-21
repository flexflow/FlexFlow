#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_VISITABLE_CORE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_VISITABLE_CORE_H

#include "utils/required_core.h"
#include "utils/tuple.h"
#include "visit_struct/visit_struct.hpp"

#define VISITABLE_STRUCT_EMPTY(STRUCT_NAME)                                    \
  namespace visit_struct {                                                     \
  namespace traits {                                                           \
                                                                               \
  template <>                                                                  \
  struct visitable<STRUCT_NAME, void> {                                        \
                                                                               \
    using this_type = STRUCT_NAME;                                             \
                                                                               \
    static VISIT_STRUCT_CONSTEXPR auto get_name() -> decltype(#STRUCT_NAME) {  \
      return #STRUCT_NAME;                                                     \
    }                                                                          \
                                                                               \
    static VISIT_STRUCT_CONSTEXPR const std::size_t field_count = 0;           \
                                                                               \
    template <typename V, typename S>                                          \
    VISIT_STRUCT_CXX14_CONSTEXPR static void apply(V &&visitor,                \
                                                   S &&struct_instance) {}     \
                                                                               \
    template <typename V, typename S1, typename S2>                            \
    VISIT_STRUCT_CXX14_CONSTEXPR static void                                   \
        apply(V &&visitor, S1 &&s1, S2 &&s2) {}                                \
                                                                               \
    template <typename V>                                                      \
    VISIT_STRUCT_CXX14_CONSTEXPR static void visit_pointers(V &&visitor) {}    \
                                                                               \
    template <typename V>                                                      \
    VISIT_STRUCT_CXX14_CONSTEXPR static void visit_types(V &&visitor) {}       \
                                                                               \
    template <typename V>                                                      \
    VISIT_STRUCT_CXX14_CONSTEXPR static void visit_accessors(V &&visitor) {}   \
                                                                               \
    struct fields_enum {                                                       \
      enum index {};                                                           \
    };                                                                         \
                                                                               \
    static VISIT_STRUCT_CONSTEXPR const bool value = true;                     \
  };                                                                           \
  }                                                                            \
  }                                                                            \
  static_assert(true, "")

namespace FlexFlow {

template <typename T>
using is_visitable = ::visit_struct::traits::is_visitable<T>;

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

template <typename T>
struct field_count : std::integral_constant<
                         std::size_t,
                         ::visit_struct::traits::visitable<T>::field_count> {};

} // namespace FlexFlow

#endif
