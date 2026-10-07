#include "task-spec/dynamic_graph/serializable_dynamic_value_attrs.h"
#include <optional>

namespace FlexFlow {

SerializableDynamicValueAttrs
    dynamic_value_attrs_to_serializable(DynamicValueAttrs const &attrs) {
  return SerializableDynamicValueAttrs{
      /*tensor_guid=*/attrs.tensor_guid,
      /*parallel_tensor_shape=*/attrs.parallel_tensor_shape,
      /*create_grad=*/attrs.create_grad,
      /*subgradient_id=*/attrs.subgradient_id,
      /*shard_coord=*/attrs.shard_coord,
      /*mapping=*/attrs.mapping,
      /*role=*/attrs.role,
  };
}

DynamicValueAttrs dynamic_value_attrs_from_serializable(
    SerializableDynamicValueAttrs const &attrs) {
  return DynamicValueAttrs{
      /*tensor_guid=*/attrs.tensor_guid,
      /*parallel_tensor_shape=*/attrs.parallel_tensor_shape,
      /*create_grad=*/attrs.create_grad,
      /*subgradient_id=*/attrs.subgradient_id,
      /*shard_coord=*/attrs.shard_coord,
      /*mapping=*/attrs.mapping,
      /*accessor=*/std::nullopt,
      /*role=*/attrs.role,
  };
}

} // namespace FlexFlow
