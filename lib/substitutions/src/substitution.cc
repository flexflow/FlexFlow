#include "substitutions/substitution.h"

namespace FlexFlow {

template <typename T>
GraphAttributeValue graph_attribute_value_op(AttrOpType op, T const &lhs, T const &rhs) {
  switch (op) {
    case AttrOpType::ADD:
        return lhs + rhs;
        break;
    case AttrOpType::SUB:
        return lhs - rhs;
        break;
    case AttrOpType::MUL:
        return lhs * rhs;
        break;
    case AttrOpType::DIV:
        return lhs / rhs;
        break;
    default:
        mk_runtime_error("Unknown attribute operator type");
  }
}

struct EvaluateGraphAttributeExpr {
  template <typename... Ts>
  GraphAttributeValue operator()(Ts... const &ts) {
    return evaluate(ts);
  }

  template <typename T>
  GraphAttributeValue evaluate(NodeAttrAccess<T> const &t) {
    Node node_in_pattern = t.node;
    Node node_in_pcg = match.nodeAssignment.at_l(node_in_pattern);
    return evaluate_attribute_expr(node_in_pcg, t.attr_expr);
  }

  template <typename T>
  GraphAttributeValue evaluate(EdgeAttrAccess<T> const &t) {
    OpenMultiDiEdge edge_in_pattern = t.edge;
    MultiDiEdge edge_in_pcg = match.edgeAssignment.at_l(edge_in_pattern);
    return evaluate_attribute_expr(edge_in_pcg, t.attr_expr);
  }

  template <typename L, typename R>
  GraphAttributeValue evaluate(AttrUnary<L, R> const &t) {
    auto lhs = (*this)(t.lhs).value();
    auto rhs = t.rhs;
    return graph_attribute_value_op(lhs, rhs);
  }

  template <typename L, typename R>
  GraphAttributeValue evaluate(AttrBinary<L, R> const &t) {
    auto lhs = (*this)(t.lhs).value();
    auto rhs = (*this)(t.rhs).value();
    return graph_attribute_value_op(lhs, rhs);
  }

  EvaluateGraphAttributeExpr(ParallelComputationGraph const &graph, DiGraphPatternMatch const &match) : graph(graph), match(match) {}

  ParallelComputationGraph const &graph;
  DiGraphPatternMatch const &match;
};

template <typename... Ts>
GraphAttributeValue evaluate_graph_attribute_expr(ParallelComputationGraph const &graph, DiGraphPatternMatch const &match, GraphAttributeExpr<Ts...> const &expr) {
  return visit(EvaluateGraphAttributeExpr(graph, match), expr);
}

Operator get_operator_attrs(ParallelComputationGraph const &graph, DiGraphPatternMatch const &match, OperatorAttrAssignment const &assignment) {
  NOT_IMPLEMENTED();
}

ParallelTensor get_parallel_tensor_attrs(ParallelComputationGraph const &graph, DiGraphPatternMatch const &match, ParallelTensorAttrAssignment const &assignment) {
  NOT_IMPLEMENTED();
}

ParallelComputationGraph apply_substitution(ParallelComputationGraph const &pcg,
                                            Substitution const &substitution,
                                            DiGraphPatternMatch const &match) {
  ParallelComputationGraph new_pcg = ParallelComputationGraph::create<UnorderedOutputLabelledMultiDiGraph>();
  bidict<Node, Node> node_mapping; // Refactor it with global nodes
  for (Node const &node : get_nodes(pcg)) {
    if (!contains_r(match.nodeAssignment)) {
      node_mapping.equate(node, new_pcg.add_node(pcg.at(node)));
    }
  }
  for (MultiDiEdge const &edge : get_edges(pcg)) {
    if (!contains_r(match.edgeAssignment)) {
      new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(edge.src), node_mapping.at_r(edge.dst), new_pcg.add_node_port(), new_pcg.add_node_port()});
    }
  }
  for (Node const &output_node : get_nodes(substitution.output_graph)) {
    Node new_node = new_pcg.add_node(get_operator_attrs(pcg, match, substitution.output_graph.at(output_node)));
    node_mapping.equate(output_node, new_node);
  }
  for (OpenMultiDiEdge const &output_edge : get_edges(substitution.output_graph)) {
    if (holds_alternative<InputMultiDiEdge>(output_edge)) {
      MultiDiEdge origin_edge = match.edgeAssignment.at_r(substitution.input_mapping.at_r(output_edge));
      new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(origin_edge.src), node_mapping.at_l(output_edge.dst), new_pcg.add_node_port(), new_pcg.add_node_port()});
    } else if (holds_alternative<OutputMultiDiEdge>(output_edge)) {
      MultiDiEdge origin_edge = match.edgeAssignment.at_r(substitution.output_mapping.at_r(output_edge));
      new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(output_edge.src), node_mapping.at_l(origin_edge.dst), new_pcg.add_node_port(), new_pcg.add_node_port()});
    } else {
      assert(holds_alternative<MultiDiEdge>(output_edge));
      new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(output_edge.src), node_mapping.at_l(output_edge.dst), new_pcg.add_node_port(), new_pcg.add_node_port()});
    }
  }
  for (MultiDiOutput const &output : get_outputs(substitution.output_graph)) {
    new_pcg.add_output(MultiDiOutput{node_mapping.at_l(output.src), new_pcg.add_node_port()}, get_parallel_tensor_attrs(pcg, match, substitution.output_graph.at(output)));
  }

  return new_pcg;
}

}