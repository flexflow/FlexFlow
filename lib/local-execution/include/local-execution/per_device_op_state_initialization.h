#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_PER_DEVICE_OP_STATE_INITIALIZATION_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_PER_DEVICE_OP_STATE_INITIALIZATION_H

#include "kernels/allocation.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/profiling_settings.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include "task-spec/global_device_id_t.dtg.h"

namespace FlexFlow {

bool no_nodes_are_initialized(DynamicOpenDataflowGraph const &g);

DynamicNodeInvocation initialize_node(DynamicNodeInvocation const &i,
                                      Allocator &allocator,
                                      device_handle_t const &device_handle,
                                      OptimizerAttrs const &optimizer_attrs,
                                      global_device_id_t device_idx);

/**
 * @brief Initialize all operators and save the per-device op state
 */
DynamicOpenDataflowGraph perform_per_device_op_state_initialization(
    DynamicOpenDataflowGraph const &,
    Allocator &allocator,
    device_handle_t const &device_handle,
    OptimizerAttrs const &optimizer_attrs,
    global_device_id_t device_idx);

} // namespace FlexFlow

#endif
