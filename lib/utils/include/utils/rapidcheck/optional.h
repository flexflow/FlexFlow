#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RAPIDCHECK_OPTIONAL_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RAPIDCHECK_OPTIONAL_H

#include <rapidcheck.h>
#include <optional>

namespace rc {

template <typename T>
struct Arbitrary<std::optional<T>> {
  static Gen<std::optional<T>> arbitrary() {
    return gen::map(
        gen::maybe(std::move(gen::arbitrary<T>())), [](Maybe<T> &&m) {
          return m ? std::optional<T>(std::move(*m)) : std::optional<T>();
        });
  }
};

} // namespace rc

#endif
