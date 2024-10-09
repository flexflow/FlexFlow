#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_GRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_UNDIRECTED_GRAPH_H

#include "utils/graph/node/graph_view.h"
#include "utils/graph/node/i_graph.h"
#include "utils/graph/node/node.dtg.h"
#include "utils/graph/node/node_query.dtg.h"

namespace FlexFlow {

struct Graph : virtual GraphView {
public:
  Graph(Graph const &) = default;

  Graph &operator=(Graph const &) = default;

  Node add_node();
  void add_node_unsafe(Node const &);
  void remove_node_unsafe(Node const &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;

  template <typename T>
  static typename std::enable_if<std::is_base_of<IGraph, T>::value, Graph>::type
      create() {
    return Graph(make_cow_ptr<T>());
  }

  using GraphView::GraphView;

private:
  IGraph const &get_ptr() const;
  IGraph &get_ptr();

  friend struct GraphInternal;
};

} // namespace FlexFlow

#endif
