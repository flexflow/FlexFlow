#ifndef _FLEXFLOW_INPUT_H
#define _FLEXFLOW_INPUT_H

#include "local-execution/op_task_invocation.h"
#include "op-attrs/ops/input.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(InputAttrs const &);

} // namespace FlexFlow

#endif
