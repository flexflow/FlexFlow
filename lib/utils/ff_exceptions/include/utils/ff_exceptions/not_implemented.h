#ifndef _FLEXFLOW_LIB_UTILS_FF_EXCEPTIONS_INCLUDE_UTILS_FF_EXCEPTIONS_NOT_IMPLEMENTED_H
#define _FLEXFLOW_LIB_UTILS_FF_EXCEPTIONS_INCLUDE_UTILS_FF_EXCEPTIONS_NOT_IMPLEMENTED_H

#include <stdexcept>

namespace FlexFlow {

#ifdef FF_REQUIRE_IMPLEMENTED
#define NOT_IMPLEMENTED() static_assert(false, "Function not yet implemented");
#else
#define NOT_IMPLEMENTED() throw ::FlexFlow::not_implemented();
#endif

class not_implemented : public std::logic_error {
public:
  not_implemented();
};

} // namespace FlexFlow

#endif
