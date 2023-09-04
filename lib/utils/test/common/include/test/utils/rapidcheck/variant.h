#ifndef _FLEXFLOW_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_RAPIDCHECK_VARIANT_H
#define _FLEXFLOW_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_RAPIDCHECK_VARIANT_H

#include "rapidcheck.h"
#include "rapidcheck/gen/Arbitrary.h"
#include "utils/type_traits.h"
#include "utils/variant.h"

namespace rc {

template <typename... Ts>
struct Arbitrary<variant<Ts...>> {
  static Gen<variant<Ts...>> arbitrary() {
    static_assert(::FlexFlow::pretty_elements_satisfy<supports_rc_arbitrary,
                                                      variant<Ts...>>::value,
                  "All fields must support arbitrary");

    /* return gen::apply( */
    /*     [](::FlexFlow::visit_as_tuple_t<T> const &t) -> T { */
    /*       return ::FlexFlow::visitable_from_tuple<T>(t); */
    /*     }, */
    /*     gen::arbitrary<::FlexFlow::visit_as_tuple_t<T>>()); */
  }
};

} // namespace rc

#endif
