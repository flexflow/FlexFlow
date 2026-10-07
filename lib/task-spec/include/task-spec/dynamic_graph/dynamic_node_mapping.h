#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_NODE_MAPPING_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_NODE_MAPPING_H

#include "task-spec/dynamic_graph/dynamic_node_mapping.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_slot.dtg.h"
#include "task-spec/dynamic_graph/parallel_tensor_mapping.dtg.h"
#include "task-spec/global_device_id_t.dtg.h"

namespace FlexFlow {

bidict<global_device_id_t, OperatorAtomicTaskShardBinding>
    dynamic_node_mapping_get_shard_bindings(DynamicNodeMapping const &);

OperatorAtomicTaskShardBinding
    dynamic_node_mapping_get_shard_binding_for_device(
        DynamicNodeMapping const &, global_device_id_t const &);

bidict<ParallelTensorSpaceCoordinate, global_device_id_t>
    dynamic_node_mapping_bindings_for_slot_name(DynamicNodeMapping const &,
                                                TensorSlotName const &);

std::set<global_device_id_t>
    target_devices_of_dynamic_node_mapping(DynamicNodeMapping const &);

} // namespace FlexFlow

#endif
