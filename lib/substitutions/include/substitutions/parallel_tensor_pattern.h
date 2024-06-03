#ifndef _FLEXFLOW_SUBSTITUTIONS_TENSOR_PATTERN_H
#define _FLEXFLOW_SUBSTITUTIONS_TENSOR_PATTERN_H

#include "attribute_expr.h"
#include "pcg/parallel_tensor.h"

namespace FlexFlow {

/**
 * @brief TensorAttributeKey is an enum class that represents the keys of the 
 * attributes of a Tensor(matrix).
 * DIM_SIZES describes the size of each dimension of the tensor for data parallelism computation
 * DIM_DEGREES describes the number of partitions along each dimension of the tensor for data parallelism computation
 */
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


/**
 * @brief evaluate_attribute_expr evaluates the attribute expression for a given ParallelTensor
 * the ParallelTensor parameter is named tensor_shape because the numerical value will only be used
 * in runtime. For the substitution phase, all that matters is the shape of the tensor.
 */
std::optional<TensorAttributeValue>
    evaluate_attribute_expr(ParallelTensor const &tensor_shape,
                            AttributeExpr<TensorAttributeKey> const &expr);

} // namespace FlexFlow

#endif
