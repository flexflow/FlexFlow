#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_I_UNDIRECTED_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_I_UNDIRECTED_GRAPH_H

#include "utils/graph/undirected/i_undirected_graph_view.h"

namespace FlexFlow {

struct IUndirectedGraph : public IUndirectedGraphView {
  virtual Node add_node() = 0;
  virtual void add_node_unsafe(Node const &) = 0;
  virtual void remove_node_unsafe(Node const &) = 0;
  virtual void add_edge(UndirectedEdge const &) = 0;
  virtual void remove_edge(UndirectedEdge const &) = 0;

  virtual std::unordered_set<Node>
      query_nodes(NodeQuery const &query) const = 0;

  virtual IUndirectedGraph *clone() const = 0;
};

} // namespace FlexFlow

#endif
