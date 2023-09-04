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

struct IsValidOperatorAttributeExprFunctor {
  GraphPattern const &graph_pattern;

  template <typename T>
  bool operator()(T const &t) const {
    return is_valid(t);
  }

  bool is_valid(OperatorAttrAccess const &t) const {
    return is_valid_operator_attribute_expr(graph_pattern->at(t.node),
                                            t.attr_expr);
  }

  bool is_valid(AttrConstant const &t) const {
    return true;
  }
};

bool is_valid_operator_attribute_expr(GraphPattern const &pattern,
                                   OperatorAttributeExpr const &expr) {
  return visit(IsValidOperatorAttributeExprFunctor{pattern}, expr);
}

bool is_valid_substitution(Substitution const &s) {
  for (Node const &node : get_nodes(s.output_graph_expr)) {
    for (OperatorAttributeExpr expr :
         values(s.output_graph_expr->at(node).assignment)) {
      if (!is_valid_operator_attribute_expr(s.input_graph, expr)) {
        return false;
      }
    }
  }
  return true;
}

struct EvaluateOperatorAttributeExpr {
  ParallelComputationGraph const &graph;
  MultiDiGraphPatternMatch const &match;

  template <typename T>
  OperatorAttributeExpr operator()(T const &t) {
    return evaluate(t);
  }

  OperatorAttributeValue evaluate(OperatorAttrAccess const &t) {
    Node node_in_pattern = t.node;
    Node node_in_pcg = match.node_assignment.at_l(node_in_pattern);
    return evaluate_attribute_expr(graph->at(node_in_pcg), t.attr_expr).value();
  }

  OperatorAttributeValue evaluate(AttrConstant const &t) {
    return t.value;
  }
};

OperatorAttributeExpr
    evaluate_graph_attribute_expr(ParallelComputationGraph const &g,
                                  MultiDiGraphPatternMatch const &match,
                                  OperatorAttributeExpr const &expr) {
  return visit(EvaluateOperatorAttributeExpr{g, match}, expr);
}

Operator get_operator_attrs(ParallelComputationGraph const &graph,
                            MultiDiGraphPatternMatch const &match,
                            OperatorAttrAssignment const &assignment) {
  NOT_IMPLEMENTED();
}

ParallelComputationGraph
    apply_substitution(ParallelComputationGraph const &pcg,
                       Substitution const &substitution,
                       MultiDiGraphPatternMatch const &match) {
  ParallelComputationGraph new_pcg =
      OutputLabelledMultiDiGraph<Operator, ParallelTensor>::create<
          UnorderedOutputLabelledMultiDiGraph<Operator, ParallelTensor>>();
  bidict<Node, Node> node_mapping; // Refactor it with global nodes
  for (Node const &node : get_nodes(pcg)) {
    if (!contains_r(match.node_assignment, node)) {
      node_mapping.equate(node, new_pcg->add_node(pcg.value().at(node)));
    }
  }
  for (MultiDiEdge const &edge : get_edges(pcg)) {
    if (!contains_r(match.edge_assignment, edge)) {
      new_pcg->add_edge(MultiDiEdge{node_mapping.at_l(edge.src),
                                   node_mapping.at_r(edge.dst),
                                   new_pcg->add_node_port(),
                                   new_pcg->add_node_port()});
    }
  }
  for (Node const &output_node : get_nodes(substitution.output_graph_expr)) {
    Node new_node = new_pcg->add_node(get_operator_attrs(
        pcg, match, substitution.output_graph_expr->at(output_node)));
    node_mapping.equate(output_node, new_node);
  }
  for (OpenMultiDiEdge const &output_edge :
       get_edges(substitution.output_graph_expr)) {
    if (holds_alternative<InputMultiDiEdge>(output_edge)) {
      InputMultiDiEdge e = get<InputMultiDiEdge>(output_edge);
      MultiDiEdge original_edge = match.edge_assignment.at_l(
          substitution.input_mapping.at_r(e));
      new_pcg->add_edge(MultiDiEdge{node_mapping.at_l(original_edge.src),
                                   node_mapping.at_l(e.dst),
                                   new_pcg->add_node_port(),
                                   new_pcg->add_node_port()});
    } else if (holds_alternative<OutputMultiDiEdge>(output_edge)) {
      OutputMultiDiEdge e = get<OutputMultiDiEdge>(output_edge);
      MultiDiEdge original_edge = match.edge_assignment.at_l(
          substitution.output_mapping.at_r(e));
      new_pcg->add_edge(MultiDiEdge{node_mapping.at_l(e.src),
                                   node_mapping.at_l(original_edge.dst),
                                   new_pcg->add_node_port(),
                                   new_pcg->add_node_port()});
    } else {
      assert(holds_alternative<MultiDiEdge>(output_edge));
      MultiDiEdge e = get<MultiDiEdge>(output_edge);
      new_pcg->add_edge(MultiDiEdge{node_mapping.at_l(e.src),
                                   node_mapping.at_l(e.dst),
                                   new_pcg->add_node_port(),
                                   new_pcg->add_node_port()});
    }
  }

  return new_pcg;
}

} // namespace FlexFlow
