#ifndef _FLEXFLOW_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_RAPIDCHECK_VARIANT_H
#define _FLEXFLOW_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_RAPIDCHECK_VARIANT_H

#include "rapidcheck.h"
#include "rapidcheck/gen/Arbitrary.h"
#include "supports_rc_arbitrary.h"
#include <variant>

namespace rc {

template <typename... Ts>
struct Arbitrary<std::variant<Ts...>> {
  static Gen<std::variant<Ts...>> arbitrary() {
    static_assert((supports_rc_arbitrary_v<Ts> && ...), "All fields must support arbitrary");

    return gen::construct<std::variant<Ts...>>(gen::oneOf(gen::arbitrary<Ts>()...));
  }
};

} // namespace rc

#endif
