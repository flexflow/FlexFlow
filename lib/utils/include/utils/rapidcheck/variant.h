#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RAPIDCHECK_VARIANT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RAPIDCHECK_VARIANT_H

#include <variant>
#include <rapidcheck.h>

namespace rc {

template <typename... Ts>
struct Arbitrary<std::variant<Ts...>> {
  static Gen<std::variant<Ts...>> arbitrary() {
    return gen::oneOf(gen::construct<std::variant<Ts...>>(gen::arbitrary<Ts>())...);
  }
};

} // namespace rc

#endif
