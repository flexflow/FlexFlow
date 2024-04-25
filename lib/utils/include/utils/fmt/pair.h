#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_PAIR_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_FMT_PAIR_H

#include <utility>
#include "fmt/format.h"
#include "utils/check_fmtable.h"

namespace FlexFlow {

template <typename L, typename R>
std::ostream &operator<<(std::ostream &s, std::pair<L, R> const &m) {
  CHECK_FMTABLE(L);
  CHECK_FMTABLE(R);

  return s << fmt::to_string(m);
}

} // namespace FlexFlow

#endif
