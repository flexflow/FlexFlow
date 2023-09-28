#ifndef _FLEXFLOW_LIB_UTILS_RAPIDCHECK_EXTRA_INCLUDE_UTILS_RAPIDCHECK_EXTRA_OPTIONAL_H
#define _FLEXFLOW_LIB_UTILS_RAPIDCHECK_EXTRA_INCLUDE_UTILS_RAPIDCHECK_EXTRA_OPTIONAL_H

#include <optional>
#include "rapidcheck.h"
#include "supports_rc_arbitrary.h"

namespace rc {

template <typename T>
struct Arbitrary<std::optional<T>> {
  static Gen<std::optional<T>> arbitrary() {
    static_assert(rc::supports_rc_arbitrary_v<T>, "Underlying type must support arbitrary");
    return gen::sizedOneOf(
        gen::just<std::optional<T>>(std::nullopt),
        gen::construct<std::optional<T>>(gen::arbitrary<T>()));
  }
};

} // namespace FlexFlow

#endif
