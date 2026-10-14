#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP3_STRICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP3_STRICT_H

#include "utils/containers/zip3.h"
#include "utils/fmt/vector.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename A, typename B, typename C>
std::vector<std::tuple<A, B, C>> zip3_strict(std::vector<A> const &as,
                                             std::vector<B> const &bs,
                                             std::vector<C> const &cs) {
  if (!(as.size() == bs.size() && bs.size() == cs.size())) {
    PANIC(fmt::format(
        "zip3_strict requires as, bs, and cs to have the same length, but "
        "received as={} (length {}), bs={} (length {}), and cs={} (length {})",
        as,
        as.size(),
        bs,
        bs.size(),
        cs,
        cs.size()));
  }

  return zip3(as, bs, cs);
}

} // namespace FlexFlow

#endif
