#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_OPTIONAL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_OPTIONAL_H

#include "fmt.h"
#include "utils/exception.h"
#include "utils/optional.decl"

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

} // namespace FlexFlow

namespace fmt {

template <typename T, typename Char>
struct formatter<
  ::std::optional<T>,
  Char,
  std::enable_if_t<!detail::has_format_as<std::optional<T>>::value>
> : formatter<std::string> {
  template <typename FormatContext>
  auto format(::std::optional<T> const &q, FormatContext &ctx)
      -> decltype(ctx.out()) {
    std::string result;
    if (q.has_value()) {
      result = fmt::to_string(q.value());
    } else {
      result = "nullopt";
    }
    return formatter<std::string>::format(result, ctx);
  }
};

} // namespace fmt

#endif
