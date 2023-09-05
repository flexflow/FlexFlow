#ifndef _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_H
#define _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_H

#include "utils/exception.decl.h"
#include "utils/fmt.h"
#include <stdexcept>

namespace FlexFlow {

template <typename... T>
std::runtime_error mk_runtime_error(fmt::format_string<T...> fmt_str,
                                    T &&...args) {
  return std::runtime_error(
      fmt::vformat(fmt_str, fmt::make_format_args(args...)));
}

} // namespace FlexFlow

#endif
