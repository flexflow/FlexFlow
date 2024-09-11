#include "utils/graph/open_dataflow_graph/algorithms/from_open_dataflow_graph_data.h"
#include "utils/graph/dataflow_graph/dataflow_output_query.h"
#include "utils/graph/node/node_query.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_edge_query.h"

namespace FlexFlow {

FromOpenDataflowGraphDataView::FromOpenDataflowGraphDataView(
    OpenDataflowGraphData const &data)
    : data(data) {}

std::unordered_set<Node>
    FromOpenDataflowGraphDataView::query_nodes(NodeQuery const &q) const {
  return apply_node_query(q, this->data.nodes);
}

std::unordered_set<OpenDataflowEdge> FromOpenDataflowGraphDataView::query_edges(
    OpenDataflowEdgeQuery const &q) const {
  return apply_open_dataflow_edge_query(q, this->data.edges);
}

std::unordered_set<DataflowOutput> FromOpenDataflowGraphDataView::query_outputs(
    DataflowOutputQuery const &q) const {
  return apply_dataflow_output_query(q, this->data.outputs);
}

std::unordered_set<DataflowGraphInput>
    FromOpenDataflowGraphDataView::get_inputs() const {
  return this->data.inputs;
}

FromOpenDataflowGraphDataView *FromOpenDataflowGraphDataView::clone() const {
  return new FromOpenDataflowGraphDataView{this->data};
}

OpenDataflowGraphView
    from_open_dataflow_graph_data(OpenDataflowGraphData const &data) {
  return OpenDataflowGraphView::create<FromOpenDataflowGraphDataView>(data);
}

} // namespace FlexFlow
