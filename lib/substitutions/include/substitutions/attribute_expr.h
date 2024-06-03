#ifndef _FLEXFLOW_SUBSTITUTIONS_CONSTRAINT_H
#define _FLEXFLOW_SUBSTITUTIONS_CONSTRAINT_H

#include "utils/variant.h"

namespace FlexFlow {

enum class ConstraintType { EQUAL };

/**
 * @struct ListIndexAccess
 * @brief Given the attribute key, retrieve the specific value stored at index i in the attribute
 * This struct will be used in EvaluateOperatorAttributeExpr and EvaluateTensorAttributeExpr, 
 * where we evaluate the expression and return the concrete value of the attribute stored at index i
 */
template <typename T>
struct ListIndexAccess {
  T attribute_key;
  req<int> index;
};

/**
 * @struct ListSize
 * @brief Given the type of an attribute, retrieve the size of the attribute
 * Specifically, for the OperatorAttributeValue, the size of the attribute is always MAX_TENSOR_DIM
 * For the TensorAttributeValue, the size of the attribute is the size of the vector that represents 
 * the specific attribute of tensor in PCG
 */
template <typename T>
struct ListSize {
  req<T> attribute_key;
};

/**
 * @struct AttributeExpr
 * @brief AttributeExpr is a representation of ways to access the attribute.
 * It can be a direct value, or a list index access, or a list size. 
 * For example, padding of a Conv2D operator will be represented as a int, 
 * and the dimension of a tensor will be represented as a vector to which
 * we can access the vector size with ListSize and access the specific value 
 * with ListIndexAccess
 */
template <typename T>
using AttributeExpr = std::variant<T, ListIndexAccess<T>, ListSize<T>>;


/**
 * @struct AttributeConstraint
 * @brief AttributeConstraint is additional constraint imposed when doing pattern matching other than 
 * just matching graph topology. Specifically, given a pattern and a graph, matching solely the attribute 
 * type is not enough as there are other factors to consider. For example, if we want to fuse two dense
 * layer, we need to match the input shape; given a dense layer, we need to make sure the input shape matches 
 * the output shape of the previous layer.
 * 
 * Given an attribute expression, attribute_expr should have a relationship with attribute_value defined by 
 * constraint_type. Currently only EQUAL is supported, meaning that the attribute_expr should be equal to 
 * attribute_value after evaluation.
 */
template <typename K, typename V>
struct AttributeConstraint {
  ConstraintType constraint_type;
  AttributeExpr<K> attribute_expr;
  V attribute_value;
};


/**
 * @struct AttributePattern
 * @brief AttributePattern is a collection of attribute constraints for pattern matching to satisfy.
 */
template <typename K, typename V>
struct AttributePattern {
  std::vector<AttributeConstraint<K, V>> attribute_constraints;
  // TODO: Revert to unordered_set once we have visitable for templates
  // std::unordered_set<AttributeConstraint<K, V>> attribute_constraints;
};

} // namespace FlexFlow

#endif
