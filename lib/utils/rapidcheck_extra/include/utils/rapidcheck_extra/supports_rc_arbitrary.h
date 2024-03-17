#ifndef _FLEXFLOW_LIB_UTILS_RAPIDCHECK_EXTRA_INCLUDE_UTILS_RAPIDCHECK_EXTRA_SUPPORTS_RC_ARBITRARY_H
#define _FLEXFLOW_LIB_UTILS_RAPIDCHECK_EXTRA_INCLUDE_UTILS_RAPIDCHECK_EXTRA_SUPPORTS_RC_ARBITRARY_H

#include "rapidcheck.h"
#include "rapidcheck/gen/Arbitrary.h"
#include <type_traits>

namespace rc {

template <typename T, typename Enable = void>
struct supports_rc_arbitrary : std::false_type {};

template <typename T>
struct supports_rc_arbitrary<T,
                             std::void_t<decltype(Arbitrary<T>::arbitrary())>>
    : std::true_type {};

template <typename T>
inline constexpr bool supports_rc_arbitrary_v = supports_rc_arbitrary<T>::value;

} // namespace rc

#endif
