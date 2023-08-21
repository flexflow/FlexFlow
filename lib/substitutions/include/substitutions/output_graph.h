#ifndef _FLEXFLOW_SUBSTITUTIONS_OUTPUT_GRAPH_H
#define _FLEXFLOW_SUBSTITUTIONS_OUTPUT_GRAPH_H

#include "utils/graph.h"

namespace FlexFlow {

// NOTE(@wmdi) I am not sure whether these should be part of attribute expr.
template <typename T>
struct NodeAttrAccess {
  Node node;
  T attr_expr;
};

template <typename T>
struct EdgeAttrAccess {
  OpenMultiDiEdge edge;
  T attr_expr;
};

enum class AttrOpType { ADD, SUB, MUL, DIV };

template <typename T>
struct AttrConstant {
  T value;
};
template <typename L, typename R>
struct AttrUnary {
  AttrOpType op_type;
  L lhs;
  R rhs;
};

template <typename L, typename R>
struct AttrBinary {
  AttrOpType op_type;
  L lhs;
  R rhs;
};

template <typename T>
using GraphAttributeExpr =
    variant<NodeAttrAccess<T>, EdgeAttrAccess<T>, AttrConstant<T>>;

template <typename L, typename R>
using GraphAttributeExpr = variant<AttrUnary<L, R>, AttrBinary<L, R>>;

using GraphAttributeValue =
    variant<int, float, bool, std::vector<int>, OperatorType, Activation>;

// NOTE(@wmdi): Not sure if it aligns with other design. Or alternatively we can
// define the assignment for each operator type.
template <typename T>
struct OperatorAttrAssignment {
  std::vector<std::pair<OperatorAttributeKey, GraphAttributeExpr<T>>>
      assignment;
};

template <typename T>
struct ParallelTensorAttrAssignment {
  std::vector<std::pair<TensorAttributeKey, GraphAttributeExpr<T>>> assignment;
};

struct OutputGraph
    : public strong_typedef<
          OutputGraph,
          OutputLabelledMultiDiGraph<OperatorAttrAssignment,
                                     ParallelTensorAttrAssignment>> {
  using strong_typedef::strong_typedef;
};

} // namespace FlexFlow

#endif
