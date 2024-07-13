#ifndef _FLEXFLOW_UTILS_INCLUDE_FMT_H
#define _FLEXFLOW_UTILS_INCLUDE_FMT_H

#include "utils/fmt.decl.h"
#include "utils/test_types.h"
#include "utils/type_traits_core.h"
#include <iomanip>
#include <unordered_set>
#include <variant>
#include <vector>

namespace FlexFlow {

template <typename T>
typename std::enable_if<delegate_ostream_operator<std::decay_t<T>>::value,
                        std::ostream &>::type
    operator<<(std::ostream &s, T t) {
  CHECK_FMTABLE(T);

  return s << fmt::to_string(t);
}

} // namespace FlexFlow

#endif
