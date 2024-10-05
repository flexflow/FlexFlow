#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ONLY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ONLY_H

#include "utils/containers/maybe_get_only.h"
#include "utils/exception.h"
#include "utils/optional.h"

namespace FlexFlow {

template <typename C>
typename C::value_type get_only(C const &c) {
  return unwrap(maybe_get_only(c), [&] {
    throw mk_runtime_error(fmt::format(
        "Encountered container with size {} in get_only", c.size()));
  });
}

} // namespace FlexFlow

#endif
