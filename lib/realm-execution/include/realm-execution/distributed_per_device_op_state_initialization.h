#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DISTRIBUTED_PER_DEVICE_OP_STATE_INITIALIZATION_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DISTRIBUTED_PER_DEVICE_OP_STATE_INITIALIZATION_H

#include "pcg/optimizer_attrs.dtg.h"
#include "realm-execution/distributed_ff_handle.h"
#include "realm-execution/per_device_op_state_backing.dtg.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/tensor_instance_backing.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

/**
 * @brief Launches tasks (using \ref spawn_per_device_op_state_init_task) to
 * create the \ref PerDeviceOpState ""s for each %GPU and packages the results
 * into a PerDeviceOpStateBacking.
 *
 * \relates PerDeviceOpStateBacking
 */
PerDeviceOpStateBacking perform_distributed_per_device_op_state_initialization(
    RealmContext &ctx,
    DynamicOpenDataflowGraph const &dg,
    TensorInstanceBacking const &tensor_instance_backing,
    DistributedFfHandle const &device_handle,
    OptimizerAttrs const &optimizer_attrs,
    Realm::Event precondition);

} // namespace FlexFlow

#endif
