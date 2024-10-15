#ifndef UTILS_GRAPH_INCLUDE_UTILS_GRAPH_DIGRAPH
#define UTILS_GRAPH_INCLUDE_UTILS_GRAPH_DIGRAPH

#include "utils/graph/cow_ptr_t.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/digraph/directed_edge.dtg.h"
#include "utils/graph/digraph/directed_edge_query.dtg.h"
#include "utils/graph/digraph/i_digraph.h"

namespace FlexFlow {

struct DiGraph : virtual DiGraphView {
public:
  using Edge = DirectedEdge;
  using EdgeQuery = DirectedEdgeQuery;

  DiGraph(DiGraph const &) = default;
  DiGraph &operator=(DiGraph const &) = default;

  Node add_node();
  void add_node_unsafe(Node const &);
  void remove_node_unsafe(Node const &);

  void add_edge(Edge const &);
  void remove_edge(Edge const &);

  std::unordered_set<Node> query_nodes(NodeQuery const &) const;
  std::unordered_set<Edge> query_edges(EdgeQuery const &) const;

  template <typename T>
  static typename std::enable_if<std::is_base_of<IDiGraph, T>::value,
                                 DiGraph>::type
      create() {
    return DiGraph(make_cow_ptr<T>());
  }

protected:
  using DiGraphView::DiGraphView;

private:
  IDiGraph &get_ptr();
  IDiGraph const &get_ptr() const;
};
CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(DiGraph);

} // namespace FlexFlow

#endif
