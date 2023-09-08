#ifndef _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_H
#define _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_H

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

std::runtime_error mk_runtime_error(std::string const &);
} // namespace FlexFlow

#endif
