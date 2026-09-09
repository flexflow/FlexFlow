#include "task-spec/dynamic_graph/dynamic_value_attrs.h"

namespace FlexFlow {

DynamicValueAttrs
    decide_dynamic_value_attrs_role(DynamicValueAttrs const &attrs,
                                    DynamicTensorRole role) {
  ASSERT(attrs.role == std::nullopt);

  DynamicValueAttrs result = attrs;
  result.role = role;

  return result;
}

DynamicValueAttrs decide_dynamic_value_attrs_subgradient_id(
    DynamicValueAttrs const &attrs, subgradient_id_t const &subgradient_id) {
  ASSERT(!attrs.subgradient_id.has_value());

  DynamicValueAttrs result = attrs;
  result.subgradient_id = subgradient_id;

  return result;
}

DynamicValueAttrs
    decide_dynamic_value_attrs_mapping(DynamicValueAttrs const &attrs,
                                       ParallelTensorMapping const &mapping) {
  ASSERT(!attrs.mapping.has_value());

  DynamicValueAttrs result = attrs;
  result.mapping = mapping;

  return result;
}

} // namespace FlexFlow
