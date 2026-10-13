#include "task-spec/dynamic_graph/serializable_dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/serializable_dynamic_value_attrs.h"
#include "utils/bidict/algorithms/unordered_bidict_transform_values.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

SerializableDynamicOpenDataflowGraph
    dynamic_open_dataflow_graph_to_serializable(
        DynamicOpenDataflowGraph const &g) {
  return SerializableDynamicOpenDataflowGraph{
      /*invocations=*/transform(g.invocations,
                                dynamic_node_invocation_to_serializable),
      /*invocation_ids=*/
      unordered_bidict_transform_values(
          g.invocation_ids, dynamic_node_invocation_to_serializable),
      /*value_ids=*/
      unordered_bidict_transform_values(g.value_ids,
                                        dynamic_value_attrs_to_serializable),
  };
}

DynamicOpenDataflowGraph dynamic_open_dataflow_graph_from_serializable(
    SerializableDynamicOpenDataflowGraph const &serializable) {
  return DynamicOpenDataflowGraph{
      /*invocations=*/transform(serializable.invocations,
                                dynamic_node_invocation_from_serializable),
      /*invocation_ids=*/
      unordered_bidict_transform_values(
          serializable.invocation_ids,
          dynamic_node_invocation_from_serializable),
      /*value_ids=*/
      unordered_bidict_transform_values(serializable.value_ids,
                                        dynamic_value_attrs_from_serializable),
  };
}

} // namespace FlexFlow
