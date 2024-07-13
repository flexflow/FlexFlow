#include "substitutions/unlabelled/pattern_matching.h"
#include "substitutions/unlabelled/match_split.h"
#include "substitutions/unlabelled/pattern_node_output.h"
#include "substitutions/unlabelled/pattern_split.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph_inputs.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph.h"
#include <memory>
#include "utils/graph/node/algorithms.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.h"
#include "utils/overload.h"
#include "substitutions/unlabelled/pattern_edge.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge.dtg.h"
#include "substitutions/unlabelled/input_pattern_edge.h" 
#include "substitutions/unlabelled/standard_pattern_edge.h" 

namespace FlexFlow {

OpenDataflowSubgraphResult subgraph_matched(OpenDataflowGraphView const &g,
                   UnlabelledDataflowGraphPatternMatch const &match) {
  std::unordered_set<Node> matched_nodes = keys(match.node_assignment.reversed());
  return get_subgraph(g, matched_nodes);
}

// bool are_dataflow_graphs_equal_under(DataflowGraphView const &l,
//                                      DataflowGraphView const &r,
//                                      bidict<Node, Node> const &matching) {
//   std::unordered_set<Node> l_nodes = get_nodes(l);
//
//   auto l_from_r = [&](Node const &r_node) {
//     return matching.at_r(r_node);
//   };
//
//   std::unordered_set<Node> l_from_r_nodes = transform(get_nodes(r), l_from_r);
//
//   if (l_nodes != l_from_r_nodes) {
//     return false;
//   }
//
//   std::unordered_set<DataflowEdge> l_edges = get_edges(l);
//   std::unordered_set<DataflowEdge> l_from_r_edges = transform(get_edges(r),
//                            [&](DataflowEdge const &r_edge) { 
//                              return DataflowEdge{
//                                DataflowOutput{
//                                  l_from_r(r_edge.src.node),
//                                  r_edge.src.idx,
//                                },
//                                DataflowInput{
//                                  l_from_r(r_edge.dst.node),
//                                  r_edge.dst.idx,
//                                }
//                              };
//                            });
//
//   if (l_edges != l_from_r_edges) {
//     return false;
//   }
//
//   return true;
// }

struct ConcreteFromPattern {
  ConcreteFromPattern(UnlabelledDataflowGraphPatternMatch const &match)
    : match(match) 
  { }

  UnlabelledDataflowGraphPatternMatch const &match;

  Node operator()(PatternNode const &n) const {
    return match.node_assignment.at_l(n);
  }

  OpenDataflowValue operator()(PatternInput const &i) const {
    return match.input_assignment.at(i);
  }

  OpenDataflowEdge operator()(InputPatternEdge const &e) const {
    return open_dataflow_edge_from_src_and_dst( 
      this->operator()(get_src_input(e)),
      DataflowInput{
        this->operator()(get_dst_node(e)),
        get_dst_idx(e),
      }
    );
  }

  DataflowEdge operator()(StandardPatternEdge const &e) const {
    return DataflowEdge{
      DataflowOutput{
        this->operator()(get_src_node(e)),
        get_src_idx(e),
      },
      DataflowInput{
        this->operator()(get_dst_node(e)),
        get_dst_idx(e),
      },
    };
  }

  OpenDataflowEdge operator()(PatternEdge const &pattern_e) const {
    return pattern_e.visit<OpenDataflowEdge>([&](auto const &e) { return OpenDataflowEdge{this->operator()(e)}; });
  }

  OpenDataflowValue operator()(PatternValue const &pattern_v) const {
    return pattern_v.visit<OpenDataflowValue>([&](auto const &v) { return OpenDataflowValue{this->operator()(v)}; });
  }

  DataflowOutput operator()(PatternNodeOutput const &o) const {
    return DataflowOutput{
      this->operator()(get_src_node(o)),
      get_idx(o),
    };
  }
};

static bool pattern_matches_subgraph_under(UnlabelledGraphPattern const &pattern,
                                    OpenDataflowGraphView const &subgraph,
                                    UnlabelledDataflowGraphPatternMatch const &match,
                                    MatchAdditionalCriterion const &additional_criterion) {
  ConcreteFromPattern concrete_from_pattern{match};

  std::unordered_set<Node> concrete_nodes = get_nodes(subgraph);
  std::unordered_set<Node> concrete_nodes_from_match = transform(get_nodes(pattern), concrete_from_pattern);

  if (concrete_nodes != concrete_nodes_from_match) {
    return false;
  }

  for (PatternNode const &pattern_node : get_nodes(pattern)) {
    if (!additional_criterion.node_criterion(pattern_node, concrete_from_pattern(pattern_node))) {
      return false;
    }
  }

  std::unordered_set<OpenDataflowEdge> concrete_edges = get_edges(subgraph);
  std::unordered_set<OpenDataflowEdge> concrete_edge_from_match = transform(get_edges(pattern), concrete_from_pattern);

  if (concrete_edges != concrete_edge_from_match) {
    return false;
  }

  std::unordered_set<OpenDataflowValue> concrete_values = get_open_dataflow_values(subgraph);
  std::unordered_set<OpenDataflowValue> concrete_values_from_match = transform(get_values(pattern), concrete_from_pattern);

  if (concrete_values != concrete_values_from_match) {
    return false;
  }

  for (PatternValue const &pattern_value : get_values(pattern)) {
    if (!additional_criterion.value_criterion(pattern_value, concrete_from_pattern(pattern_value))) {
      return false;
    }
  }

  return true;
}

bool unlabelled_pattern_does_match(
    UnlabelledGraphPattern const &pattern,
    OpenDataflowGraphView const &graph,
    UnlabelledDataflowGraphPatternMatch const &match,
    MatchAdditionalCriterion const &additional_criterion) {

  OpenDataflowGraphView matched_subgraph = subgraph_matched(matched_subgraph, match).graph;

  assert (keys(match.node_assignment) == get_nodes(pattern));
  assert (keys(match.node_assignment.reversed()) == get_nodes(matched_subgraph));

  return pattern_matches_subgraph_under(pattern, matched_subgraph, match, additional_criterion);
}

} // namespace FlexFlow
