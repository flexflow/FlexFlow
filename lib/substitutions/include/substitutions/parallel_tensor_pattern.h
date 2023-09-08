#ifndef _FLEXFLOW_SUBSTITUTIONS_TENSOR_PATTERN_H
#define _FLEXFLOW_SUBSTITUTIONS_TENSOR_PATTERN_H

#include "attribute_expr.h"
#include "pcg/parallel_tensor.h"

namespace FlexFlow {

enum class TensorAttributeKey { DIM_SIZES, DIM_DEGREES };

using TensorAttributeValue = variant<int, std::vector<int>>;

using TensorAttributeConstraint =
    AttributeConstraint<TensorAttributeKey, TensorAttributeValue>;

using ParallelTensorPattern =
    AttributePattern<TensorAttributeKey, TensorAttributeValue>;

optional<TensorAttributeValue>
    evaluate_attribute_expr(ParallelTensor const &tensor_shape,
                            AttributeExpr<TensorAttributeKey> const &expr);

} // namespace FlexFlow

#endif
