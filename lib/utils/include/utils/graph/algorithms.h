#ifndef _FLEXFLOW_UTILS_GRAPH_ALGORITHMS_H
#define _FLEXFLOW_UTILS_GRAPH_ALGORITHMS_H

#include "node.h"
#include "multidigraph.h"
#include "digraph.h"
#include "undirected.h"
#include <vector>
#include <unordered_map>

namespace FlexFlow {
namespace utils {
namespace graph {

std::vector<Node> add_nodes(IGraph &, int);
std::unordered_set<Node> get_nodes(IGraphView const &);

std::unordered_set<multidigraph::Edge> get_edges(IMultiDiGraphView const &);
std::unordered_set<digraph::Edge> get_edges(IDiGraphView const &);
std::unordered_set<undirected::Edge> get_edges(IUndirectedGraphView const &);

std::unordered_set<multidigraph::Edge> get_incoming_edges(IMultiDiGraphView const &, std::unordered_set<Node> const &);
std::unordered_set<digraph::Edge> get_incoming_edges(IDiGraphView const &, std::unordered_set<Node> const &);
std::unordered_set<undirected::Edge> get_incoming_edges(IUndirectedGraphView const &, std::unordered_set<Node> const &);

std::unordered_set<multidigraph::Edge> get_outgoing_edges(IMultiDiGraphView const &, std::unordered_set<Node> const &);
std::unordered_set<digraph::Edge> get_outgoing_edges(IDiGraphView const &, std::unordered_set<Node> const &);
std::unordered_set<undirected::Edge> get_outgoing_edges(IUndirectedGraphView const &, std::unordered_set<Node> const &);

std::unordered_map<Node, std::unordered_set<Node>> get_predecessors(IMultiDiGraphView const &, std::unordered_set<Node> const &);
std::unordered_map<Node, std::unordered_set<Node>> get_predecessors(IDiGraphView const &, std::unordered_set<Node> const &);

bool is_acyclic(IMultiDiGraphView const &, std::unordered_set<Node> const &);
bool is_acyclic(IDiGraphView const &);

std::vector<Node> topo_sort(IMultiDiGraphView const &);
std::vector<Node> topo_sort(IDiGraphView const &);

std::unordered_map<Node, std::unordered_set<std::size_t>> dominators(IMultiDiGraphView const &);

struct generic_unchecked_dfs_iterator {
  using iterator_category = std::forward_iterator_tag;    
  using difference_type   = std::size_t;
  using value_type        = Node;
  using pointer           = Node const *; 
  using reference         = Node const &;

  generic_unchecked_dfs_iterator(IDiGraphView const &g, std::vector<Node> const &, std::unordered_set<Node> seen);
  generic_unchecked_dfs_iterator(IDiGraphView const &g, std::unordered_set<Node> const &);
  
  reference operator*() const;
  pointer operator->();

  // Prefix increment
  generic_unchecked_dfs_iterator& operator++();

  // Postfix increment
  generic_unchecked_dfs_iterator operator++(int);

  bool operator==(generic_unchecked_dfs_iterator const &other) const;
  bool operator!=(generic_unchecked_dfs_iterator const &other) const;   
private:
  std::vector<Node> stack;
  std::unordered_set<Node> seen;
  IDiGraphView const *graph;
};

generic_unchecked_dfs_iterator dfs_start(IDiGraphView const &, std::unordered_set<Node> &starting_points);
generic_unchecked_dfs_iterator dfs_end(IDiGraphView const &);

std::vector<Node> dfs_ordering(IDiGraphView const &, std::unordered_set<Node> const &starting_points);

}
}
}

#endif 
