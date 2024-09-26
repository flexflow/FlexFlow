#ifndef _FLEXFLOW_CONV_2D_H
#define _FLEXFLOW_CONV_2D_H

#include "local-execution/op_task_invocation.h"
#include "local-execution/sim_environment.h"
#include "op-attrs/ops/conv_2d_attrs.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(Conv2DAttrs const &);

TaskImplFunction get_conv_2d_init_task_impl();
TaskImplFunction get_conv_2d_fwd_task_impl();
TaskImplFunction get_conv_2d_bwd_task_impl();

OpTaskSignature get_conv_2d_init_signature();
OpTaskSignature get_conv_2d_fwd_signature();
OpTaskSignature get_conv_2d_bwd_signature();

OpTaskInvocation init(Conv2DAttrs const &);
OpTaskInvocation forward(Conv2DAttrs const &);
OpTaskInvocation backward(Conv2DAttrs const &);

} // namespace FlexFlow

#endif
