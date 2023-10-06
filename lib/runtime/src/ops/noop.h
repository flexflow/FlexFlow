#ifndef _FLEXFLOW_NOOP_H
#define _FLEXFLOW_NOOP_H

#include "op-attrs/ops/input.h"
#include "op-attrs/ops/noop.h"
#include "task_spec/op_task_invocation.h"

namespace FlexFlow {

std::optional<OpTaskInvocation> init(NoopAttrs const &);
std::optional<OpTaskInvocation> forward(NoopAttrs const &);
std::optional<OpTaskInvocation> backward(NoopAttrs const &);
} // namespace FlexFlow

#endif
