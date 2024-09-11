#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OUTPUT_GRAPH_MATERIALIZE_OPERATOR_FROM_ATTRS_MAP_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OUTPUT_GRAPH_MATERIALIZE_OPERATOR_FROM_ATTRS_MAP_H

#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "substitutions/operator_pattern/operator_attribute_key.dtg.h"
#include "substitutions/operator_pattern/operator_attribute_value.dtg.h"

namespace FlexFlow {

PCGOperatorAttrs materialize_operator_from_attrs_map(
    std::unordered_map<OperatorAttributeKey, OperatorAttributeValue> const &);

} // namespace FlexFlow

#endif
