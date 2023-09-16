#ifndef _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_H
#define _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_H

#include <stdexcept>

namespace FlexFlow {

std::runtime_error mk_runtime_error(std::string const &);

} // namespace FlexFlow

#endif
