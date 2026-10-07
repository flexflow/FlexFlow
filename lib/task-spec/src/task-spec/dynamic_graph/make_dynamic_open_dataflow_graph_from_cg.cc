#include "task-spec/dynamic_graph/make_dynamic_open_dataflow_graph_from_cg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/computation_graph.h"
#include "pcg/tensor_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_layer_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/map_from_unordered.h"
#include <map>
#include <optional>
#include <utility>

namespace FlexFlow {

DynamicOpenDataflowGraph
    make_dynamic_open_dataflow_graph_from_cg(ComputationGraph const &cg) {
  DynamicOpenDataflowGraph result = make_empty_dynamic_open_dataflow_graph();

  for (auto const &[layer, attrs] : get_layer_attrs_mapping(cg)) {
    DynamicNodeAttrs result_attrs{
        /*task_type=*/std::nullopt,
        /*device_coord=*/std::nullopt,
        /*mapping=*/std::nullopt,
        /*op_attrs=*/
        TrainingOperationAttrs{
            pcg_op_attrs_from_compgraph_op_attrs(attrs.op_attrs)},
        /*pcg_layer_guid=*/dynamic_layer_guid_t{layer},
        /*per_device_op_state=*/std::nullopt,
    };

    std::map<DynamicTensorSlot, DynamicValueAttrs> result_inputs = transform(
        get_incoming_tensors(cg, layer),
        [&](TensorSlotName const &slot_name, tensor_guid_t const &tensor) {
          TensorAttrs attrs = get_tensor_attrs(cg, tensor);
          return std::pair<DynamicTensorSlot, DynamicValueAttrs>{
              DynamicTensorSlot{
                  /*slot_name=*/slot_name,
                  /*slot_tensor_role=*/std::nullopt,
                  /*task_shard=*/std::nullopt,
              },
              DynamicValueAttrs{
                  /*tensor_guid=*/dynamic_tensor_guid_t{tensor},
                  /*parallel_tensor_shape=*/lift_to_parallel(attrs.shape),
                  /*create_grad=*/(attrs.create_grad == CreateGrad::YES),
                  /*subgradient_id=*/std::nullopt,
                  /*shard_coord=*/std::nullopt,
                  /*mapping=*/std::nullopt,
                  /*accessor=*/std::nullopt,
                  /*role=*/std::nullopt,
              },
          };
        });

    std::map<DynamicTensorSlot, DynamicValueAttrs> result_outputs = transform(
        get_outgoing_tensors(cg, layer),
        [&](TensorSlotName const &slot_name, tensor_guid_t const &tensor) {
          TensorAttrs attrs = get_tensor_attrs(cg, tensor);
          return std::pair<DynamicTensorSlot, DynamicValueAttrs>{
              DynamicTensorSlot{
                  /*slot_name=*/slot_name,
                  /*slot_tensor_role=*/std::nullopt,
                  /*task_shard=*/std::nullopt,
              },
              DynamicValueAttrs{
                  /*tensor_guid=*/dynamic_tensor_guid_t{tensor},
                  /*parallel_tensor_shape=*/lift_to_parallel(attrs.shape),
                  /*create_grad=*/(attrs.create_grad == CreateGrad::YES),
                  /*subgradient_id=*/std::nullopt,
                  /*shard_coord=*/std::nullopt,
                  /*mapping=*/std::nullopt,
                  /*accessor=*/std::nullopt,
                  /*role=*/std::nullopt,
              },
          };
        });

    result.invocations.emplace(result_inputs, result_attrs, result_outputs);
  }

  return result;
}

} // namespace FlexFlow
