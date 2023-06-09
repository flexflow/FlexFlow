#ifndef _FLEXFLOW_UTILS_INCLUDE_EXPECTED_H
#define _FLEXFLOW_UTILS_INCLUDE_EXPECTED_H

#include "tl/expected.hpp"
#include "utils/fmt.h"
#include <string>

namespace FlexFlow {

using namespace tl;

template <typename... Args>
unexpected<std::string> error_msg(Args &&...args) {
  return make_unexpected(fmt::format(std::forward<Args>(args)...));
}

} // namespace FlexFlow

#endif
