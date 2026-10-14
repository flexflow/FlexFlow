#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_STRICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_STRICT_H

#include "utils/containers/zip.h"
#include "utils/fmt/vector.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename L, typename R>
std::vector<std::pair<L, R>> zip_strict(std::vector<L> const &lhs,
                                        std::vector<R> const &rhs) {
  ASSERT(lhs.size() == rhs.size(),
         "zip_strict requires lhs and rhs to have the same length",
         lhs,
         rhs);

  return zip(lhs, rhs);
}

} // namespace FlexFlow

#endif
