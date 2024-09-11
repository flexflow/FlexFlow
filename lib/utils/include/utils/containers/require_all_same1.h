#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_ALL_SAME1_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_ALL_SAME1_H

#include <fmt/format.h>
#include <tl/expected.hpp>

namespace FlexFlow {

template <typename C, typename T = typename C::value_type>
tl::expected<T, std::string> require_all_same1(C const &c) {
  if (c.empty()) {
    return tl::unexpected(fmt::format(
        "require_all_same1 expected non-empty container, but received {}", c));
  }

  T const &first = *c.cbegin();
  for (T const &v : c) {
    if (v != first) {
      return tl::unexpected(fmt::format("require_all_same1 found non-same "
                                        "elements {} and {} in containers {}",
                                        first,
                                        v,
                                        c));
    }
  }
  return first;
}

} // namespace FlexFlow

#endif
