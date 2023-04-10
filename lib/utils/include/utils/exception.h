#ifndef _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_H
#define _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_H

#include "utils/fmt.h"

namespace FlexFlow {

template <typename ...T>
std::runtime_error mk_runtime_error(fmt::format_string<T...> fmt, T &&... args) {
  return std::runtime_error(fmt::format(fmt, std::forward<T>(args)...));
}

}

#endif
