#ifndef _FLEXFLOW_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_RAPIDCHECK_TRAITS_H
#define _FLEXFLOW_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_RAPIDCHECK_TRAITS_H

#include "rapidcheck.h"
#include "rapidcheck/gen/Arbitrary.h"
#include "utils/type_traits.h"

namespace rc {

template <typename T, typename Enable = void>
struct supports_rc_arbitrary : std::false_type {};

template <typename T>
struct supports_rc_arbitrary<T,
                             ::FlexFlow::void_t<decltype(gen::arbitrary<T>())>>
    : std::true_type {};

} // namespace rc

#endif
