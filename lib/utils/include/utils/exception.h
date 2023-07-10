#ifndef _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_H
#define _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_H

#include "utils/fmt.h"
#include <stdexcept>

namespace FlexFlow {

#ifdef FF_REQUIRE_IMPLEMENTED
#define NOT_IMPLEMENTED() static_assert(false, "Function not yet implemented");
#else
#define NOT_IMPLEMENTED() throw not_implemented();
#endif

class not_implemented : public std::logic_error {
public:
  not_implemented() : std::logic_error("Function not yet implemented"){};
};

template <typename... T>
std::runtime_error mk_runtime_error(fmt::format_string<T...> fmt_str,
                                    T &&...args) {
  return std::runtime_error(
      fmt::vformat(fmt_str, fmt::make_format_args(args...)));
}

} // namespace FlexFlow

#endif
