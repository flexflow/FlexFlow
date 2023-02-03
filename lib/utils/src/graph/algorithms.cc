#include "utils/graph/algorithms.h"
#include "utils/graph/conversions.h"

namespace FlexFlow {
namespace utils {
namespace graph {

std::unordered_set<Node> get_nodes(IMultiDiGraphView const &g) {
  return g.query_nodes({});
}

std::unordered_set<Node> get_nodes(IDiGraphView const &g) {
  return g.query_nodes({});
}

std::unordered_set<Node> get_nodes(IUndirectedGraphView const &g) {
  return g.query_nodes({});
}

std::unordered_set<multidigraph::Edge> get_edges(IMultiDiGraphView const &g) {
  return g.query_edges({});
}

std::unordered_set<digraph::Edge> get_edges(IDiGraphView const &g) {
  return g.query_edges({});
}

std::unordered_set<undirected::Edge> get_edges(IUndirectedGraphView const &g) {
  return g.query_edges({});
}

std::unordered_set<multidigraph::Edge> get_incoming_edges(IMultiDiGraphView const &g, std::unordered_set<Node> const &dsts) {
  return g.query_edges(multidigraph::EdgeQuery::all().with_dst_nodes(dsts));
}

std::unordered_set<digraph::Edge> get_incoming_edges(IDiGraphView const &g, std::unordered_set<Node> const &dsts) {
  auto multidigraph_view = unsafe_view_as_multidigraph(g);
  std::unordered_set<digraph::Edge> result;
  for (multidigraph::Edge const &e : get_incoming_edges(multidigraph_view, dsts)) {
    result.insert(to_directed_edge(e));
  }
  return result;
}

std::unordered_map<Node, std::unordered_set<multidigraph::Node>> get_predecessors(IMultiDiGraphView const &g, std::unordered_set<Node> const &nodes) {
  std::unordered_map<Node, std::unordered_set<multidigraph::Node>> predecessors;
  for (Node const &n : nodes) {
    predecessors[n];
  }
  for (multidigraph::Edge const &e : get_incoming_edges(g, nodes)) {
    predecessors.at(e.dst).insert(e.src);
  }
  return predecessors;
}

/* bool is_acyclic(IDiGraphView const &) { */
  
/* } */

bool is_acyclic(IMultiDiGraph const &g) {
  auto digraph_view = unsafe_view_as_digraph(g);
  return is_acyclic(digraph_view);
}

}
}
}

/* std::vector<Node> topo_sort(IMultiDiGraphView const &g) { */
/*   return */ 
/* } */
