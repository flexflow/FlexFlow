#ifndef _FLEXFLOW_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_OPTIMIZER_H_
#define _FLEXFLOW_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_OPTIMIZER_H_

#include "local-execution/non_graph_tensor_guid_t.dtg.h"
#include "local-execution/task_impl_function.dtg.h"
#include "local-execution/task_invocation.dtg.h"
#include "local-execution/task_signature.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "pcg/optimizers/adam_optimizer_attrs.dtg.h"
#include "pcg/optimizers/sgd_optimizer_attrs.dtg.h"

namespace FlexFlow {

TaskSignature get_update_signature(OptimizerAttrs const &);
TaskInvocation get_update_invocation(
    OptimizerAttrs const &,
    tensor_guid_t const &weight,
    std::vector<non_graph_tensor_guid_t> const &grad_buffer_tensors);
TaskImplFunction get_update_task_impl(OptimizerAttrs const &);

TaskSignature get_sgd_update_signature();
TaskInvocation sgd_update(SGDOptimizerAttrs const &,
                          tensor_guid_t const &weight,
                          non_graph_tensor_guid_t const &sgd_v);
TaskImplFunction get_sgd_update_task_impl();

TaskSignature get_adam_update_signature();
TaskInvocation adam_update(AdamOptimizerAttrs const &,
                           tensor_guid_t const &weight,
                           non_graph_tensor_guid_t const &adam_v,
                           non_graph_tensor_guid_t const &adam_m);
TaskImplFunction get_adam_update_task_impl();

} // namespace FlexFlow

#endif
