#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_OPTIONAL_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_OPTIONAL_H

#include <optional>
#include <fmt/format.h>
#include "utils/check_fmtable.h"

namespace FlexFlow {

template <typename T>
std::ostream &operator<<(std::ostream &s, std::optional<T> const &t) {
  CHECK_FMTABLE(T);

  return s << fmt::to_string(t);
}

} // namespace FlexFlow

#endif
