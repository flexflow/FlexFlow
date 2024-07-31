#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_OPTIONAL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_OPTIONAL_H

#include "utils/exception.h"
#include "utils/fmt/optional.h"
#include <rapidcheck.h>

namespace FlexFlow {

template <typename T, typename F>
T const &unwrap(std::optional<T> const &o, F const &f) {
  if (o.has_value()) {
    return o.value();
  } else {
    f();
    throw mk_runtime_error("Failure in unwrap");
  }
}

template <typename T>
T const &assert_unwrap(std::optional<T> const &o) {
  assert(o.has_value());
  return o.value();
}

template <typename F, typename T>
std::optional<std::invoke_result_t<F, T>> transform(std::optional<T> const &o,
                                                    F &&f) {
  using Return = std::invoke_result_t<F, T>;
  if (o.has_value()) {
    Return r = f(o.value());
    return std::optional<Return>{r};
  } else {
    return std::nullopt;
  }
}

} // namespace FlexFlow

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
