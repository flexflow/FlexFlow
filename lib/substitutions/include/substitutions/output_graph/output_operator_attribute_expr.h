#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OUTPUT_GRAPH_OUTPUT_OPERATOR_ATTRIBUTE_EXPR_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OUTPUT_GRAPH_OUTPUT_OPERATOR_ATTRIBUTE_EXPR_H

#include "output_operator_attribute_expr.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_attrs.dtg.h"

namespace FlexFlow {

OperatorAttributeValue evaluate_output_operator_attribute_expr(OutputOperatorAttributeExpr const &,
                                                               std::unordered_map<PatternNode, PCGOperatorAttrs> const &node_match);

} // namespace FlexFlow

#endif
