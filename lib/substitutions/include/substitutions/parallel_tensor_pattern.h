#ifndef _FLEXFLOW_SUBSTITUTIONS_TENSOR_PATTERN_H
#define _FLEXFLOW_SUBSTITUTIONS_TENSOR_PATTERN_H

#include "attribute_expr.h"
#include "pcg/parallel_tensor.h"

namespace FlexFlow {

/**
 * @brief TensorAttributeKey is an enum class that represents the keys of the attributes of a Tensor(matrix).
 * DIM_SIZES describes the length along each dimension of the tensor
 * DIM_DEGREES describes the number of partitions along each dimension of the tensor for data parallelism computation
 */
enum class TensorAttributeKey { DIM_SIZES, DIM_DEGREES };

using TensorAttributeValue = variant<int, std::vector<int>>;

using TensorAttributeConstraint =
    AttributeConstraint<TensorAttributeKey, TensorAttributeValue>;

using ParallelTensorPattern =
    AttributePattern<TensorAttributeKey, TensorAttributeValue>;

/**
 * @brief evaluate_attribute_expr evaluates the attribute expression for a given ParallelTensor
 * 
 * @param tensor_shape, which describes the attributes of a ParallelTensor
 * @param expr, which describes the specific attribute expression to be evaluated
 * @return optional<TensorAttributeValue> 
 */
optional<TensorAttributeValue>
    evaluate_attribute_expr(ParallelTensor const &tensor_shape,
                            AttributeExpr<TensorAttributeKey> const &expr);

} // namespace FlexFlow

#endif
