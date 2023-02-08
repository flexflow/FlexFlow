#include "utils/graph/algorithms.h"
#include "utils/graph/conversions.h"
#include <queue>
#include <algorithm>
#include <iostream>

namespace FlexFlow {
namespace utils {
namespace graph {

std::vector<Node> add_nodes(IGraph &g, int num_nodes) {
  std::vector<Node> nodes;
  std::generate_n(std::back_inserter(nodes), num_nodes, [&g]() { return g.add_node(); });
  return nodes;
}

std::unordered_set<Node> get_nodes(IGraphView const &g) {
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
  return to_directed_edges(get_incoming_edges(multidigraph_view, dsts));
}

std::unordered_set<multidigraph::Edge> get_outgoing_edges(IMultiDiGraphView const &g, std::unordered_set<Node> const &srcs) {
  return g.query_edges(multidigraph::EdgeQuery::all().with_src_nodes(srcs));
}

std::unordered_set<digraph::Edge> get_outgoing_edges(IDiGraphView const &g, std::unordered_set<Node> const &dsts) {
  auto multidigraph_view = unsafe_view_as_multidigraph(g);
  return to_directed_edges(get_outgoing_edges(multidigraph_view, dsts));
}

std::unordered_map<Node, std::unordered_set<Node>> get_predecessors(IDiGraphView const &g, std::unordered_set<Node> const &nodes) {
  std::unordered_map<Node, std::unordered_set<Node>> predecessors;
  for (Node const &n : nodes) {
    predecessors[n];
  }
  for (digraph::Edge const &e : get_incoming_edges(g, nodes)) {
    predecessors.at(e.dst).insert(e.src);
  }
  return predecessors;
}

std::unordered_map<Node, std::unordered_set<Node>> get_predecessors(IMultiDiGraphView const &g, std::unordered_set<Node> const &nodes) {
  return get_predecessors(unsafe_view_as_digraph(g), nodes);
}

using gudi = generic_unchecked_dfs_iterator;

gudi::generic_unchecked_dfs_iterator(IDiGraphView const &g, std::vector<Node> const &stack, std::unordered_set<Node> seen)
  : stack(stack), graph(&g), seen(seen)
{ }

gudi::generic_unchecked_dfs_iterator(IDiGraphView const &g, std::unordered_set<Node> const &starting_points) 
  : graph(&g)
{
  for (Node const &n : starting_points) {
    this->stack.push_back(n);
  }
}

gudi::reference gudi::operator*() const { return this->stack.back(); }
gudi::pointer gudi::operator->() { return &this->operator*(); }
gudi& gudi::operator++() {
  Node const last = this->operator*();
  std::cout << "[ ";
  for (auto const &thing : this->stack) {
     std::cout << thing << " ";
  }
  std::cout << "]" << std::endl;
  this->stack.pop_back();
  if (this->seen.find(last) == this->seen.end()) {
    std::unordered_set<digraph::Edge> outgoing = get_outgoing_edges(*graph, {last});
    std::cout << "outgoing = { ";
    for (auto const &thing : outgoing) {
       std::cout << thing << " ";
    }
    std::cout << "}" << std::endl;
    this->seen.insert(last);
    for (digraph::Edge const &e : outgoing) {
      stack.push_back(e.dst);
    }
  }
  return *this; 
}
gudi gudi::operator++(int) {
  auto tmp = *this; 
  ++(*this); 
  return tmp; 
}
bool gudi::operator==(gudi const &other) const {
 return this->seen == other.seen; 
}
bool gudi::operator!=(gudi const &other) const {
 return this->seen != other.seen; 
}

gudi dfs_start(IDiGraphView const &g, std::unordered_set<Node> const &starting_points) {
  return gudi(g, starting_points);
}
gudi dfs_end(IDiGraphView const &g) {
  return gudi(g, {}, get_nodes(g));
}

std::vector<Node> dfs_ordering(IDiGraphView const &g, std::unordered_set<Node> const &starting_points) {
  auto cur = dfs_start(g, starting_points);
  auto end = dfs_end(g);

  return {cur, end};
}

/* bool is_acyclic(IDiGraphView const &) { */
      
/* } */

/* bool is_acyclic(IMultiDiGraph const &g) { */
/*   auto digraph_view = unsafe_view_as_digraph(g); */
/*   return is_acyclic(digraph_view); */
/* } */

}
}
}

/* std::vector<Node> topo_sort(IMultiDiGraphView const &g) { */
/*   return */ 
/* } */
