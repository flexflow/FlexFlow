#include "utils/graph/open_dataflow_graph/algorithms/permute_input_ids.h"
#include "utils/containers/transform.h"
#include "utils/overload.h"

namespace FlexFlow {

struct PermuteInputIdsView final : public IOpenDataflowGraphView {  
public:
  PermuteInputIdsView(OpenDataflowGraphView const &g,
                      bidict<NewDataflowGraphInput, DataflowGraphInput> const &input_mapping)
    : g(g), input_mapping(input_mapping) { }

  std::unordered_set<Node> query_nodes(NodeQuery const &new_query) const override {
    return this->g.query_nodes(new_query);
  }

  std::unordered_set<OpenDataflowEdge> query_edges(OpenDataflowEdgeQuery const &new_query) const override {
    OpenDataflowEdgeQuery old_query = OpenDataflowEdgeQuery{
      DataflowInputEdgeQuery{
        this->old_input_query_set_from_new(new_query.input_edge_query.srcs),
        new_query.input_edge_query.dst_nodes,
        new_query.input_edge_query.dst_idxs,
      },
      new_query.standard_edge_query,
    };

    return transform(this->g.query_edges(old_query),
                     [&](OpenDataflowEdge const &old_edge) { 
                       return this->new_open_dataflow_edge_from_old(old_edge);
                     });
  }

  std::unordered_set<DataflowOutput> query_outputs(DataflowOutputQuery const &new_query) const override {
    return this->g.query_outputs(new_query);
  }

  std::unordered_set<DataflowGraphInput> get_inputs() const override {
    return transform(this->g.get_inputs(), [&](DataflowGraphInput const &old_input) { return this->new_input_from_old(old_input); });
  }

  PermuteInputIdsView *clone() const override {
    return new PermuteInputIdsView{
      this->g,
      this->input_mapping,
    };
  }
private:
  query_set<DataflowGraphInput> old_input_query_set_from_new(query_set<DataflowGraphInput> const &new_query_set) const {
    if (!is_matchall(new_query_set)) {
      std::unordered_set<DataflowGraphInput> old_nodes = transform(allowed_values(new_query_set),
                                                     [&](DataflowGraphInput const &new_input) { return this->old_input_from_new(new_input); });
      return query_set<DataflowGraphInput>{old_nodes};
    } else {
      return new_query_set;
    }
  }

  OpenDataflowEdge new_open_dataflow_edge_from_old(OpenDataflowEdge const &old_edge) const {
    return old_edge.visit<OpenDataflowEdge>(overload {
      [&](DataflowInputEdge const &old) {
        return OpenDataflowEdge{
          DataflowInputEdge{
            this->new_input_from_old(old.src),
            old.dst,
          },
        };
      },
      [&](DataflowEdge const &old) {
        return OpenDataflowEdge{old};
      },
    });
  }

  DataflowGraphInput new_input_from_old(DataflowGraphInput const &old_input) const {
    return this->input_mapping.at_r(old_input).raw_input;
  }

  DataflowGraphInput old_input_from_new(DataflowGraphInput const &new_input) const {
    return this->input_mapping.at_l(NewDataflowGraphInput{new_input});
  }

private:
  OpenDataflowGraphView g;
  bidict<NewDataflowGraphInput, DataflowGraphInput> input_mapping;
};

OpenDataflowGraphView permute_input_ids(OpenDataflowGraphView const &g,
                                        bidict<NewDataflowGraphInput, DataflowGraphInput> const &input_mapping) {
  return OpenDataflowGraphView::create<PermuteInputIdsView>(g, input_mapping);
}

} // namespace FlexFlow
