#ifndef _FLEXFLOW_NOOP_H
#define _FLEXFLOW_NOOP_H

#include "local-execution/op_task_invocation.h"
#include "op-attrs/ops/noop_attrs.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(NoopAttrs const &);

} // namespace FlexFlow

#endif
