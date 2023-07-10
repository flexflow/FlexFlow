#ifndef _FLEXFLOW_RUNTIME_SRC_OP_MANAGER_H
#define _FLEXFLOW_RUNTIME_SRC_OP_MANAGER_H

#include "operator.h"

namespace FlexFlow {

struct OpManager {
  template <typename T, typename... Args>
  Op *create(Args &&...args) {
    return new T(
  }
};

} // namespace FlexFlow

#endif
