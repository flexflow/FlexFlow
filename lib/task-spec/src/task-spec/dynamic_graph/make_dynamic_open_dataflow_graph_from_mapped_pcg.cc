#include "task-spec/dynamic_graph/make_dynamic_open_dataflow_graph_from_mapped_pcg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "pcg/parallel_computation_graph/parallel_tensor_attrs.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_use_t.h"
#include "task-spec/dynamic_graph/dynamic_layer_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "utils/bidict/algorithms/bidict_unordered_set_of.h"
#include "utils/bidict/algorithms/merge_disjoint_bidicts.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_keys_and_values.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/transform_pairs.h"
#include <map>
#include <optional>
#include <utility>

namespace FlexFlow {

DynamicNodeInvocation make_dynamic_node_invocation_from_mapped(
    MappedParallelLayerInvocationInfo const &invocation_info,
    DeviceType device_type) {
  DynamicNodeAttrs result_attrs{
      /*task_type=*/std::nullopt,
      /*device_ids=*/std::nullopt,
      /*mapping=*/
      DynamicNodeMapping{
          /*op_task_group=*/invocation_info.layer_info.mapping,
          /*device_type=*/device_type,
      },
      /*op_attrs=*/
      TrainingOperationAttrs{invocation_info.layer_info.attrs.op_attrs},
      /*pcg_layer_guid=*/dynamic_layer_guid_t{invocation_info.layer_info.guid},
      /*per_device_op_state=*/std::nullopt,
  };

  auto lift_kv_pair = [&](TensorSlotName slot_name,
                          ParallelTensorInfo const &tensor)
      -> std::pair<DynamicTensorSlot, DynamicValueAttrs> {
    return {
        DynamicTensorSlot{
            /*slot_name=*/slot_name,
            /*slot_tensor_role=*/std::nullopt,
            /*task_shard=*/std::nullopt,
        },
        DynamicValueAttrs{
            /*tensor_guid=*/dynamic_tensor_guid_t{tensor.guid},
            /*parallel_tensor_shape=*/tensor.attrs.shape,
            /*create_grad=*/(tensor.attrs.create_grad == CreateGrad::YES),
            /*subgradient_id=*/std::nullopt,
            /*shard_coord=*/std::nullopt,
            /*mapping=*/std::nullopt,
            /*accessor=*/std::nullopt,
            /*role=*/std::nullopt,
        },
    };
  };

  std::map<DynamicTensorSlot, DynamicValueAttrs> result_inputs =
      transform(invocation_info.incoming, lift_kv_pair);

  std::map<DynamicTensorSlot, DynamicValueAttrs> result_outputs =
      transform(invocation_info.outgoing, lift_kv_pair);

  DynamicNodeInvocation invocation = DynamicNodeInvocation{
      /*inputs=*/result_inputs,
      /*node_attrs=*/result_attrs,
      /*outputs=*/result_outputs,
  };

  return invocation;
}

DynamicOpenDataflowGraph make_dynamic_open_dataflow_graph_from_mapped_pcg(
    MappedParallelComputationGraph const &mpcg, DeviceType device_type) {

  return dynamic_open_dataflow_graph_from_invocation_set(
      transform(mpcg_get_invocation_set(mpcg),
                [&](MappedParallelLayerInvocationInfo const &mpcg_invocation)
                    -> DynamicNodeInvocation {
                  return make_dynamic_node_invocation_from_mapped(
                      mpcg_invocation, device_type);
                }));
}

} // namespace FlexFlow
