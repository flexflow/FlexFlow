#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph.h"
#include "utils/containers/enumerate_vector.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_value.dtg.h"
#include "utils/overload.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

struct OpenDataflowSubgraph final : public IOpenDataflowGraphView {
  OpenDataflowSubgraph(OpenDataflowGraphView const &full_graph, 
                       std::unordered_set<Node> const &subgraph_nodes,
                       bidict<OpenDataflowValue, DataflowGraphInput> const &full_graph_values_to_subgraph_inputs,
                       std::vector<DataflowGraphInput> const &subgraph_inputs)
    : full_graph(full_graph),
      subgraph_nodes(subgraph_nodes),
      full_graph_values_to_subgraph_inputs(full_graph_values_to_subgraph_inputs),
      subgraph_inputs(subgraph_inputs)
  { 
    assert(is_subseteq_of(this->subgraph_nodes, get_nodes(full_graph)));
    assert(without_order(subgraph_inputs) == without_order(values(full_graph_values_to_subgraph_inputs)));
  }

  std::unordered_set<Node> query_nodes(NodeQuery const &q) const override {
    return intersection(this->full_graph.query_nodes(q), this->subgraph_nodes);
  }

  std::unordered_set<OpenDataflowEdge> query_edges(OpenDataflowEdgeQuery const &q) const override {
    std::unordered_set<OpenDataflowEdge> result;
    for (OpenDataflowEdge const &open_e : this->full_graph.query_edges(q)) {
      open_e.visit<void>(overload {
        [&](DataflowEdge const &e) {
          bool contains_src = contains(this->subgraph_nodes, e.src.node);
          bool contains_dst = contains(this->subgraph_nodes, e.dst.node);
          if (contains_src && contains_dst) {
            result.insert(OpenDataflowEdge{e});
          } else if (contains_dst && !contains_src) {
            result.insert(OpenDataflowEdge{DataflowInputEdge{this->full_graph_values_to_subgraph_inputs.at_l(OpenDataflowValue{e.src}), e.dst}});
          }
        },
        [&](DataflowInputEdge const &e) {
          if (contains(this->subgraph_nodes, e.dst.node)) {
            result.insert(OpenDataflowEdge{DataflowInputEdge{this->full_graph_values_to_subgraph_inputs.at_l(OpenDataflowValue{e.src}), e.dst}});
          }
        }
      });
    }
    return result;
  }
  
  std::unordered_set<DataflowOutput> query_outputs(DataflowOutputQuery const &q) const override {
    return filter(this->full_graph.query_outputs(q),
                  [&](DataflowOutput const &o) { 
                    return contains(this->subgraph_nodes, o.node);
                  });
  }

  std::vector<DataflowGraphInput> get_inputs() const override {
    return this->subgraph_inputs;
  };

  OpenDataflowSubgraph *clone() const override {
    return new OpenDataflowSubgraph{
      this->full_graph,
      this->subgraph_nodes,
      this->full_graph_values_to_subgraph_inputs,
      this->subgraph_inputs,
    };
  }
private:
  OpenDataflowGraphView full_graph;
  std::unordered_set<Node> subgraph_nodes;
  bidict<OpenDataflowValue, DataflowGraphInput> full_graph_values_to_subgraph_inputs;
  std::vector<DataflowGraphInput> subgraph_inputs;
};


OpenDataflowGraphView get_subgraph(OpenDataflowGraphView const &g,
                                   std::unordered_set<Node> const &subgraph_nodes,
                                   std::vector<OpenDataflowValue> const &input_ordering) {
  std::vector<DataflowGraphInput> subgraph_inputs;
  bidict<OpenDataflowValue, DataflowGraphInput> full_graph_values_to_subgraph_inputs;
  for (auto const &[idx, full_graph_value] : enumerate_vector(input_ordering)) {
    DataflowGraphInput subgraph_input = DataflowGraphInput{idx};  
    subgraph_inputs.push_back(subgraph_input);
    full_graph_values_to_subgraph_inputs.equate({full_graph_value, subgraph_input});
  }

  return OpenDataflowGraphView::create<OpenDataflowSubgraph>(
    g, 
    subgraph_nodes,
    subgraph_inputs,
    input_ordering);
}

} // namespace FlexFlow
