#ifndef _FLEXFLOW_SUBSTITUTIONS_TENSOR_PATTERN_H
#define _FLEXFLOW_SUBSTITUTIONS_TENSOR_PATTERN_H

#include "constraint.h"

namespace FlexFlow {

enum class TensorDimensionAttribute { SIZE, DEGREE };

struct TensorNumDimensionsConstraint {
  int value;
};

struct TensorDimensionAttributeConstraint {
  TensorDimensionAttribute attribute;
  int index;
};

enum class TensorAttributeKey { DIM_SIZES, DIM_DEGREES };

using TensorAttributeValue = variant<int, std::vector<int>>;

using TensorAttributeConstraint =
    AttributeConstraint<TensorAttributeKey, TensorAttributeValue>;

struct ParallelTensorPattern {
  std::unordered_set<TensorAttributeConstraint> attribute_constraints;
};

optional<TensorAttributeValue>
    evaluate_attribute_expr(ParallelTensor const &tensor_shape,
                            AttributeExpr<TensorAttributeKey> const &expr);

} // namespace FlexFlow

#endif
