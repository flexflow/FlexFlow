#ifndef _FLEXFLOW_UTILS_TEST_COMMON_INCLUDE_UTILS_RAPIDCHECK_VISITABLE_H
#define _FLEXFLOW_UTILS_TEST_COMMON_INCLUDE_UTILS_RAPIDCHECK_VISITABLE_H

#include "rapidcheck.h"
#include "rapidcheck/gen/Arbitrary.h"
#include "utils/strong_typedef.h"
#include "utils/type_traits.h"
#include "utils/visitable.h"

namespace rc {

template <typename T, typename Enable = void>
struct supports_rc_arbitrary : std::false_type {};

template <typename T>
struct supports_rc_arbitrary<T,
                             ::FlexFlow::void_t<decltype(gen::arbitrary<T>())>>
    : std::true_type {};

/* template <typename T> */
/* struct supports_rc_arbitrary<T, ::FlexFlow::void_t<Arbitrary<T>>> :
 * std::true_type { }; */

template <typename T, typename Enable = void>
struct is_strong_typedef : std::false_type {};

template <typename T>
struct is_strong_typedef<T,
                         ::FlexFlow::void_t<::FlexFlow::underlying_type_t<T>>>
    : std::true_type {};

template <typename Tag>
struct Arbitrary<Tag, ::FlexFlow::enable_if_t<is_strong_typedef<Tag>::value>> {
  static Gen<Tag> arbitrary() {
    return gen::construct<Tag>(
        gen::arbitrary<::FlexFlow::underlying_type_t<Tag>>());
  }
};

template <typename T>
struct Arbitrary<T,
                 ::FlexFlow::enable_if_t<::FlexFlow::is_visitable<T>::value>> {
  static Gen<T> arbitrary() {
    static_assert(::FlexFlow::is_visitable<T>::value, "Type must be visitable");
    static_assert(::FlexFlow::elements_satisfy<supports_rc_arbitrary, T>::value,
                  "All fields must support arbitrary");

    return gen::apply(
        [](::FlexFlow::visit_as_tuple_t<T> const &t) -> T {
          return ::FlexFlow::visitable_from_tuple<T>(t);
        },
        gen::arbitrary<::FlexFlow::visit_as_tuple_t<T>>());
  }
};

} // namespace rc

#endif
