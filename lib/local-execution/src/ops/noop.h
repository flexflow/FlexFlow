#ifndef _FLEXFLOW_NOOP_H
#define _FLEXFLOW_NOOP_H

#include "local-execution/op_task_invocation.h"
#include "op-attrs/ops/input.h"
#include "op-attrs/ops/noop.h"
#include "op-attrs/ops/weight_attrs.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(InputAttrs const &);
std::vector<task_id_t> get_task_ids(NoopAttrs const &);
std::vector<task_id_t> get_task_ids(WeightAttrs const &);

std::optional<OpTaskInvocation> init(NoopAttrs const &);
std::optional<OpTaskInvocation> forward(NoopAttrs const &);
std::optional<OpTaskInvocation> backward(NoopAttrs const &);
} // namespace FlexFlow

#endif
