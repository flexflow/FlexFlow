#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NOT_IMPLEMENTED_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_NOT_IMPLEMENTED_H

#include "utils/fmt.h"
#include <fmt/format.h>
#include <libassert/assert.hpp>

namespace FlexFlow {

#ifdef FF_REQUIRE_IMPLEMENTED
#define NOT_IMPLEMENTED()                                                      \
  static_assert(false,                                                         \
                "Function " __FUNC__ " not yet implemented " __FILE__          \
                ":" __LINE__);
#else
#define NOT_IMPLEMENTED() PANIC("Not implemented");
#endif

class not_implemented : public std::logic_error {
public:
  not_implemented(std::string const &function_name,
                  std::string const &file_name,
                  int line);
};

} // namespace FlexFlow

#endif
