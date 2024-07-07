#include "substitutions/substitution.h"

namespace FlexFlow {

/* struct AddMappedEdgeFunctor { */
/*   bidict<Node, Node> const &node_mapping; */
/*   SubParallelComputationGraph &new_pcg; */

/*   template <typename T> */
/*   void operator()(T const &t) { */
/*     return add_mapped_edge(t); */
/*   } */

/*   void add_mapped_edge(InputMultiDiEdge const &e) { */
/*     new_pcg.add_edge(InputMultiDiEdge{ */
/*         node_mapping.at_l(e.dst), new_pcg.add_node_port(), e.uid}); */
/*   } */

/*   void add_mapped_edge(OutputMultiDiEdge const &e) { */
/*     new_pcg.add_edge(OutputMultiDiEdge{ */
/*         node_mapping.at_l(e.src), new_pcg.add_node_port(), e.uid}); */
/*   } */

/*   void add_mapped_edge(MultiDiEdge const &e) { */
/*     new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(e.dst), */
/*                                  new_pcg.add_node_port(), */
/*                                  node_mapping.at_l(e.src), */
/*                                  new_pcg.add_node_port()}); */
/*   } */
/* }; */

/* struct AddNewEdgeFunctor { */
/*   SubParallelComputationGraph const &old_pcg; */
/*   SubParallelComputationGraph &new_pcg; */
/*   MultiDiGraphPatternMatch const &match; */
/*   bidict<Node, Node> node_mapping; */

/*   template <typename TO, typename TN> */
/*   void operator()(TO const &old_edge, TN const &new_edge) { */
/*     return add_new_edge(old_edge, new_edge); */
/*   } */

/*   void add_new_edge(InputMultiDiEdge const &old_edge, */
/*                     InputMultiDiEdge const &new_edge) { */
/*     new_pcg.add_edge(InputMultiDiEdge{node_mapping.at_l(new_edge.dst), */
/*                                       new_pcg.add_node_port(), */
/*                                       old_edge.uid}); */
/*   } */

/*   void add_new_edge(MultiDiEdge const &old_edge, */
/*                     InputMultiDiEdge const &new_edge) { */
/*     new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(new_edge.dst), */
/*                                  new_pcg.add_node_port(), */
/*                                  node_mapping.at_l(old_edge.src), */
/*                                  new_pcg.add_node_port()}); */
/*   } */

/*   void add_new_edge(OutputMultiDiEdge const &old_edge, */
/*                     OutputMultiDiEdge const &new_edge) { */
/*     new_pcg.add_edge(OutputMultiDiEdge{node_mapping.at_l(new_edge.src), */
/*                                        new_pcg.add_node_port(), */
/*                                        old_edge.uid}); */
/*   } */

/*   void add_new_edge(MultiDiEdge const &old_edge, */
/*                     OutputMultiDiEdge const &new_edge) { */
/*     new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(old_edge.dst), */
/*                                  new_pcg.add_node_port(), */
/*                                  node_mapping.at_l(new_edge.src), */
/*                                  new_pcg.add_node_port()}); */
/*   } */

/*   void add_new_edge(InputMultiDiEdge const &, OutputMultiDiEdge const &) { */
/*     assert(false); */
/*   } */

/*   void add_new_edge(OpenMultiDiEdge const &, MultiDiEdge const &) { */
/*     assert(false); */
/*   } */

/*   void add_new_edge(OutputMultiDiEdge const &, InputMultiDiEdge const &) { */
/*     assert(false); */
/*   } */
/* }; */

/* SubParallelComputationGraph */
/*     apply_substitution(SubParallelComputationGraph const &pcg, */
/*                        Substitution const &substitution, */
/*                        MultiDiGraphPatternMatch const &match) { */
/*   SubParallelComputationGraph new_pcg = */
/*       OutputLabelledOpenMultiDiGraph<Operator, ParallelTensor>::template
 * create< */
/*           UnorderedOutputLabelledOpenMultiDiGraph<Operator,
 * ParallelTensor>>(); */
/*   bidict<Node, Node> node_mapping; // Refactor it with global nodes */
/*   for (Node const &node : get_nodes(pcg)) { */
/*     if (!contains_r(match.node_assignment, node)) { */
/*       node_mapping.equate(node, new_pcg.add_node(pcg.at(node))); */
/*     } */
/*   } */
/*   for (OpenMultiDiEdge const &edge : get_edges(pcg)) { */
/*     if (!contains_r(match.edge_assignment, edge)) { */
/*       visit(AddMappedEdgeFunctor{node_mapping, new_pcg}, edge); */
/*     } */
/*   } */
/*   for (Node const &output_node : */
/*        get_nodes(substitution.output_graph_expr.value())) { */
/*     Operator new_op = get_operator_attrs( */
/*         pcg, match, substitution.output_graph_expr.value().at(output_node));
 */
/*     Node new_node = new_pcg.add_node(new_op); */
/*     node_mapping.equate(output_node, new_node); */
/*   } */
/*   for (OpenMultiDiEdge const &output_edge : */
/*        get_edges(substitution.output_graph_expr.value())) { */
/*     if (std::holds_alternative<InputMultiDiEdge>(output_edge)) { */
/*       InputMultiDiEdge e = std::get<InputMultiDiEdge>(output_edge); */
/*       OpenMultiDiEdge original_edge = */
/*           match.edge_assignment.at_l(substitution.input_mapping.at_r(e)); */
/*       visit(AddNewEdgeFunctor{pcg, new_pcg, match, node_mapping}, */
/*             original_edge, */
/*             output_edge); */
/*     } else if (std::holds_alternative<OutputMultiDiEdge>(output_edge)) { */
/*       OutputMultiDiEdge e = std::get<OutputMultiDiEdge>(output_edge); */
/*       OpenMultiDiEdge original_edge = */
/*           match.edge_assignment.at_l(substitution.output_mapping.at_r(e)); */
/*       visit(AddNewEdgeFunctor{pcg, new_pcg, match, node_mapping}, */
/*             original_edge, */
/*             output_edge); */
/*     } else { */
/*       assert(std::holds_alternative<MultiDiEdge>(output_edge)); */
/*       MultiDiEdge e = std::get<MultiDiEdge>(output_edge); */
/*       new_pcg.add_edge(MultiDiEdge{node_mapping.at_l(e.dst), */
/*                                    new_pcg.add_node_port(), */
/*                                    node_mapping.at_l(e.src), */
/*                                    new_pcg.add_node_port()}); */
/*     } */
/*   } */

/*   return new_pcg; */
/* } */

bool is_valid_substitution(Substitution const &) {
  NOT_IMPLEMENTED();
}

SubParallelComputationGraph
    apply_substitution(SubParallelComputationGraph const &,
                       Substitution const &,
                       UnlabelledDataflowGraphPatternMatch const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
