#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OPERATOR_PATTERN_OPERATOR_ATTRIBUTE_EXPR_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_OPERATOR_PATTERN_OPERATOR_ATTRIBUTE_EXPR_H

#include "pcg/parallel_computation_graph/parallel_layer_attrs.dtg.h"
#include "substitutions/operator_pattern/operator_attribute_expr.dtg.h"
#include "substitutions/operator_pattern/operator_attribute_value.dtg.h"
#include <optional>

namespace FlexFlow {

std::optional<OperatorAttributeValue>
    evaluate_attribute_expr(OperatorAttributeExpr const &expr,
                            PCGOperatorAttrs const &attrs);
} // namespace FlexFlow

#endif
