#ifndef _FLEXFLOW_SUBSTITUTIONS_OUTPUT_GRAPH_H
#define _FLEXFLOW_SUBSTITUTIONS_OUTPUT_GRAPH_H

#include "utils/graph.h"

namespace FlexFlow {

// NOTE(@wmdi) I am not sure whether these should be part of attribute expr.

/**
 * @struct OperatorAttrAccess
 * @brief OperatorAttrAccess consists of a node and an expression attr_expr 
 * on the attributes of the operator associated with the node. The value of a 
 * NodeAttrAccess instance is the value of attr_expr evaluated on the operator 
 * associated with the node.
 */
struct OperatorAttrAccess {
  Node node;
  AttributeExpr<OperatorAttributeKey> attr_expr;
};

/**
 * @struct AttrConstant
 * @brief AttrConstant is a constant value that is used as an attribute expression.
 */
struct AttrConstant {
  OperatorAttributeValue value;
};

/**
 * @brief OperatorAttributeExpr is a access to the attribute of an operator and can be
 * evaluated to a concrete value. OperatorAttributeExpr is used at substitution phase. 
 * It will be evaluated and used to create new operator with the evaluated value.
 */
using OperatorAttributeExpr = variant<OperatorAttrAccess, AttrConstant>;

/**
 * @brief OperatorAttrAssignment is a collection of OperatorAttributeKey and 
 * GraphAttributeExpr pairs for a single operator. It defines how the attributes 
 * of a single operator is calculated from the input graph. A pair 
 * {operator_attribute_key, graph_attribute_expr} in the collection means the value 
 * of graph_attribute_expr is assigned to the attribute named operator_attribute_key 
 * of the operator.
 */
struct OperatorAttrAssignment {
  std::unordered_map<OperatorAttributeKey, OperatorAttributeExpr> assignments;
};

/**
 * @brief An OutputGraphExpr is defined as an open graph with node label 
 * OperatorAttrAssignment and output label ParallelTensorAttrAssignment, which 
 * defines how the operator attributes and the parallel tensor attributes of the 
 * output graph are derived from the input graph.
 */
struct OutputGraphExpr
    : public strong_typedef<
          OutputGraphExpr,
          NodeLabelledOpenMultiDiGraph<OperatorAttrAssignment>> {
  using strong_typedef::strong_typedef;
};

} // namespace FlexFlow

#endif
