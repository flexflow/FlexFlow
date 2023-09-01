#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_OP_TENSOR_SPEC_REF_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_OP_TENSOR_SPEC_REF_H

#include "op_task_signature.h"

namespace FlexFlow {

struct OpTensorSpec {
  TensorRole role;
  req<int> idx;
};
FF_VISITABLE_STRUCT(OpTensorSpec, role, idx);

OpTensorSpec input_tensor(int);
OpTensorSpec output_tensor(int);
OpTensorSpec weight_tensor(int);

} // namespace FlexFlow

#endif
