#ifndef _FLEXFLOW_UTILS_INCLUDE_EXPECTED_H
#define _FLEXFLOW_UTILS_INCLUDE_EXPECTED_H

#include "tl/expected.hpp"
#include <string>
#include "utils/fmt.h"

namespace FlexFlow {

using namespace tl;

template <typename T, typename ...Args>
unexpected<std::string> error_msg(Args &&... args) {
  return make_unexpected(fmt::format(std::forward<Args>(args)...));
}

}

#endif
