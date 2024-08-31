#include "utils/graph/dataflow_graph/algorithms/view_as_open_dataflow_graph.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

ViewDataflowGraphAsOpen::ViewDataflowGraphAsOpen(DataflowGraphView const &g)
  : g(g)
{ }

std::unordered_set<Node> ViewDataflowGraphAsOpen::query_nodes(NodeQuery const &q) const {
  return this->g.query_nodes(q);
}

std::unordered_set<OpenDataflowEdge> ViewDataflowGraphAsOpen::query_edges(OpenDataflowEdgeQuery const &q) const {
  std::unordered_set<DataflowEdge> closed_edges = this->g.query_edges(q.standard_edge_query);

  return transform(closed_edges, [](DataflowEdge const &e) { return OpenDataflowEdge{e}; });
}

std::unordered_set<DataflowOutput> ViewDataflowGraphAsOpen::query_outputs(DataflowOutputQuery const &q) const {
  return this->g.query_outputs(q);
}

std::unordered_set<DataflowGraphInput> ViewDataflowGraphAsOpen::get_inputs() const {
  return {};
}

ViewDataflowGraphAsOpen *ViewDataflowGraphAsOpen::clone() const {
  return new ViewDataflowGraphAsOpen{this->g};
}

OpenDataflowGraphView view_as_open_dataflow_graph(DataflowGraphView const &g) {
  return OpenDataflowGraphView::create<ViewDataflowGraphAsOpen>(g);
}

} // namespace FlexFlow
