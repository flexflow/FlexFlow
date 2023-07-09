#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_OP_ARG_REF_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_OP_ARG_REF_H

#include "arg_ref.h"

namespace FlexFlow {

enum class OpArgRefType { PER_DEVICE_OP_STATE };

template <typename T>
using OpArgRef = ArgRef<OpArgRefType, T>;

using OpArgRefSpec = ArgRefSpec<OpArgRefType>;

template <typename T>
OpArgRef<T> per_device_op_state() {
  return {OpArgRefType::PER_DEVICE_OP_STATE};
}

}

#endif
