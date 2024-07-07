#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_TENSOR_ATTRIBUTE_EXPR_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_TENSOR_PATTERN_TENSOR_ATTRIBUTE_EXPR_H

#include "pcg/parallel_computation_graph/parallel_tensor_attrs.dtg.h"
#include "substitutions/tensor_pattern/tensor_attribute_expr.dtg.h"
#include "substitutions/tensor_pattern/tensor_attribute_value.dtg.h"

namespace FlexFlow {

TensorAttributeValue evaluate_attribute_expr(ParallelTensorAttrs const &attrs,
                                             TensorAttributeExpr const &expr);

} // namespace FlexFlow

#endif
