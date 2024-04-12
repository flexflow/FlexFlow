#ifndef _FLEXFLOW_SUBSTITUTIONS_TENSOR_PATTERN_H
#define _FLEXFLOW_SUBSTITUTIONS_TENSOR_PATTERN_H

#include "attribute_expr.h"
#include "pcg/parallel_tensor.h"

namespace FlexFlow {

enum class TensorAttributeKey { DIM_SIZES, DIM_DEGREES };


/**
 * @brief DIM_SIZES and DIM_DEGREES are represented by 
 * a vector of ints that is listed as corresponding dimension
 */
using TensorAttributeValue = std::variant<int, std::vector<int>>;

/**
 * @brief TensorAttributeConstraint is an instance of AttributeConstraint that
 * defines the contraint a tensor should satisfy when doing pattern matching.
 */
using TensorAttributeConstraint =
    AttributeConstraint<TensorAttributeKey, TensorAttributeValue>;

/**
 * @brief ParallelTensor is an instance of OperatorAttributeExpr that represents
 * a set of constraints pattern matching should satisfy.
 */
using ParallelTensorPattern =
    AttributePattern<TensorAttributeKey, TensorAttributeValue>;

optional<TensorAttributeValue>
    evaluate_attribute_expr(ParallelTensor const &tensor_shape,
                            AttributeExpr<TensorAttributeKey> const &expr);

} // namespace FlexFlow

#endif
