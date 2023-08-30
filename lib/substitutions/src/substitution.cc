#include "substitutions/substitution.h"

namespace FlexFlow {

std::unordered_set<AttributeExpr<OperatorAttributeKey>>
    get_valid_operator_attribute_exprs(OperatorPattern const &pattern) {
  NOT_IMPLEMENTED();
}

bool is_valid_operator_attribute_expr(
    OperatorPattern const &pattern,
    AttributeExpr<OperatorAttributeKey> const &expr) {
  return contains(get_valid_operator_attribute_exprs(pattern), expr);
}

struct IsValidGraphAttributeExprFunctor {
  GraphPattern const &graph_pattern;

  template <typename T>
  bool operator()(T const &t) const {
    return is_valid(t);
  }

  bool is_valid(NodeAttrAccess const &t) const {
    return is_valid_operator_attribute_expr(graph_pattern.value().at(t.node),
                                            t.attr_expr);
  }

  bool is_valid(EdgeAttrAccess const &t) const {
    NOT_IMPLEMENTED();
  }

  bool is_valid(AttrConstant const &t) const {
    return true;
  }
};

bool is_valid_graph_attribute_expr(GraphPattern const &pattern,
                                   GraphAttributeExpr const &expr) {
  return visit(IsValidGraphAttributeExprFunctor{pattern}, expr);
}

bool is_valid_substitution(Substitution const &s) {
  for (Node const &node : get_nodes(s.output_graph_expr)) {
    for (GraphAttributeExpr expr :
         values(s.output_graph_expr.value().at(node).assignment)) {
      if (!is_valid_graph_attribute_expr(s.input_graph, expr)) {
        return false;
      }
    }
  }
  return true;
}

struct EvaluateGraphAttributeExpr {
  ParallelComputationGraph const &graph;
  MultiDiGraphPatternMatch const &match;

  template <typename T>
  GraphAttributeValue operator()(T const &t) {
    return evaluate(t);
  }

  GraphAttributeValue evaluate(NodeAttrAccess const &t) {
    Node node_in_pattern = t.node;
    Node node_in_pcg = match.node_assignment.at_l(node_in_pattern);
    return widen<GraphAttributeValue>(
        evaluate_attribute_expr(graph.at(node_in_pcg), t.attr_expr).value());
  }

  GraphAttributeValue evaluate(EdgeAttrAccess const &t) {
    OpenMultiDiEdge output_in_pattern = t.edge;
    MultiDiEdge output_in_pcg = match.edge_assignment.at_l(output_in_pattern);
    return widen<GraphAttributeValue>(
        evaluate_attribute_expr(graph.at(output_in_pcg), t.attr_expr).value());
  }
};

GraphAttributeValue
    evaluate_graph_attribute_expr(ParallelComputationGraph const &g,
                                  MultiDiGraphPatternMatch const &match,
                                  GraphAttributeExpr const &expr) {
  return visit(EvaluateGraphAttributeExpr{g, match}, expr);
}

Operator get_operator_attrs(ParallelComputationGraph const &graph,
                            MultiDiGraphPatternMatch const &match,
                            OperatorAttrAssignment const &assignment) {
  NOT_IMPLEMENTED();
}

ParallelTensor
    get_parallel_tensor_attrs(ParallelComputationGraph const &graph,
                              MultiDiGraphPatternMatch const &match,
                              ParallelTensorAttrAssignment const &assignment) {
  NOT_IMPLEMENTED();
}

ParallelComputationGraph
    apply_substitution(ParallelComputationGraph const &pcg,
                       Substitution const &substitution,
                       MultiDiGraphPatternMatch const &match) {
  ParallelComputationGraph new_pcg =
      ParallelComputationGraph::create<UnorderedOutputLabelledMultiDiGraph>();
  bidict<Node, Node> node_mapping; // Refactor it with global nodes
  for (Node const &node : get_nodes(pcg)) {
    if (!contains_r(match.node_assignment, node)) {
      node_mapping.equate(node, new_pcg.add_node(pcg.at(node)));
    }
  }
  for (MultiDiEdge const &edge : get_edges(pcg)) {
    if (!contains_r(match.edge_assignment, edge)) {
      new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(edge.src),
                                   node_mapping.at_r(edge.dst),
                                   new_pcg.add_node_port(),
                                   new_pcg.add_node_port()});
    }
  }
  for (Node const &output_node : get_nodes(substitution.output_graph_expr)) {
    Node new_node = new_pcg.add_node(get_operator_attrs(
        pcg, match, substitution.output_graph_expr.at(output_node)));
    node_mapping.equate(output_node, new_node);
  }
  for (OpenMultiDiEdge const &output_edge :
       get_edges(substitution.output_graph_expr)) {
    if (holds_alternative<InputMultiDiEdge>(output_edge)) {
      MultiDiEdge origin_edge = match.edge_assignment.at_r(
          substitution.input_mapping.at_r(output_edge));
      new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(origin_edge.src),
                                   node_mapping.at_l(output_edge.dst),
                                   new_pcg.add_node_port(),
                                   new_pcg.add_node_port()});
    } else if (holds_alternative<OutputMultiDiEdge>(output_edge)) {
      MultiDiEdge origin_edge = match.edge_assignment.at_r(
          substitution.output_mapping.at_r(output_edge));
      new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(output_edge.src),
                                   node_mapping.at_l(origin_edge.dst),
                                   new_pcg.add_node_port(),
                                   new_pcg.add_node_port()});
    } else {
      assert(holds_alternative<MultiDiEdge>(output_edge));
      new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(output_edge.src),
                                   node_mapping.at_l(output_edge.dst),
                                   new_pcg.add_node_port(),
                                   new_pcg.add_node_port()});
    }
  }
  for (MultiDiOutput const &output :
       get_outputs(substitution.output_graph_expr)) {
    new_pcg.add_output(
        MultiDiOutput{node_mapping.at_l(output.src), new_pcg.add_node_port()},
        get_parallel_tensor_attrs(
            pcg, match, substitution.output_graph_expr.at(output)));
  }

  return new_pcg;
}

} // namespace FlexFlow
