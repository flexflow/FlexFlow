#include "utils/graph/open_dataflow_graph/algorithms/permute_node_ids.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/transform.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node_query.h"
#include "utils/graph/node/node_source.h"
#include "utils/graph/query_set.h"
#include "utils/overload.h"
#include "utils/bidict/bidict.h"

namespace FlexFlow {

struct PermuteNodeIdsView final : public IOpenDataflowGraphView {
  PermuteNodeIdsView(OpenDataflowGraphView const &g, 
                     bidict<NewNode, Node> const &new_node_tofrom_old_node)
    : g(g), new_node_tofrom_old_node(new_node_tofrom_old_node)
  { }

  std::unordered_set<Node> query_nodes(NodeQuery const &new_query) const override {
    NodeQuery old_query = NodeQuery{
      this->old_node_query_set_from_new(new_query.nodes),
    };
  
    return transform(this->g.query_nodes(old_query), 
                     [&](Node const &old_node) { return this->new_node_from_old(old_node); });
  }

  std::unordered_set<OpenDataflowEdge> query_edges(OpenDataflowEdgeQuery const &new_query) const override {
    OpenDataflowEdgeQuery old_query = OpenDataflowEdgeQuery{
      DataflowInputEdgeQuery{
        new_query.input_edge_query.srcs,
        this->old_node_query_set_from_new(new_query.input_edge_query.dst_nodes),
        new_query.input_edge_query.dst_idxs,
      },
      DataflowEdgeQuery{
        this->old_node_query_set_from_new(new_query.standard_edge_query.src_nodes),
        new_query.standard_edge_query.src_idxs,
        this->old_node_query_set_from_new(new_query.standard_edge_query.dst_nodes),
        new_query.standard_edge_query.dst_idxs,
      },
    };

    return transform(this->g.query_edges(old_query),
                     [&](OpenDataflowEdge const &old_edge) { 
                       return this->new_open_dataflow_edge_from_old(old_edge);
                     });
  }

  std::unordered_set<DataflowOutput> query_outputs(DataflowOutputQuery const &new_query) const override {
    DataflowOutputQuery old_query = DataflowOutputQuery{
      this->old_node_query_set_from_new(new_query.nodes),
      new_query.output_idxs,
    };

    return transform(this->g.query_outputs(old_query),
                     [&](DataflowOutput const &old_output) {
                       return this->new_dataflow_output_from_old(old_output);
                     });
  }

  std::unordered_set<DataflowGraphInput> get_inputs() const override {
    return this->g.get_inputs();
  }

  PermuteNodeIdsView *clone() const override {
    return new PermuteNodeIdsView(
      this->g,
      this->new_node_tofrom_old_node
    );
  }

private:
  query_set<Node> old_node_query_set_from_new(query_set<Node> const &new_node_query) const {
    if (!is_matchall(new_node_query)) {
      std::unordered_set<Node> old_nodes = transform(allowed_values(new_node_query),
                                                     [&](Node const &new_node) { return this->old_node_from_new(new_node); });
      return query_set<Node>{old_nodes};
    } else {
      return new_node_query;
    }
  }

  OpenDataflowEdge new_open_dataflow_edge_from_old(OpenDataflowEdge const &old_edge) const {
    return old_edge.visit<OpenDataflowEdge>(overload {
      [&](DataflowInputEdge const &old) {
        return OpenDataflowEdge{
          DataflowInputEdge{
            old.src,
            this->new_dataflow_input_from_old(old.dst),
          }
        };
      },
      [&](DataflowEdge const &old) {
        return OpenDataflowEdge{
          DataflowEdge{
            this->new_dataflow_output_from_old(old.src),
            this->new_dataflow_input_from_old(old.dst),
          },
        };
      },
    });
  }

  DataflowOutput new_dataflow_output_from_old(DataflowOutput const &old_output) const {
    return DataflowOutput{
      this->new_node_from_old(old_output.node),
      old_output.idx,
    };
  }

  DataflowInput new_dataflow_input_from_old(DataflowInput const &old_input) const {
    return DataflowInput{
      this->new_node_from_old(old_input.node),
      old_input.idx,
    };
  }

  Node old_node_from_new(Node const &new_node) const {
    return this->new_node_tofrom_old_node.at_l(NewNode{new_node});
  }

  Node new_node_from_old(Node const &old_node) const {
    return this->new_node_tofrom_old_node.at_r(old_node).raw_node;
  }

  OpenDataflowGraphView g;
  bidict<NewNode, Node> new_node_tofrom_old_node;
};

OpenDataflowGraphView permute_node_ids(OpenDataflowGraphView const &g, 
                                      bidict<NewNode, Node> const &new_node_tofrom_old_node) {
  return OpenDataflowGraphView::create<PermuteNodeIdsView>(g, new_node_tofrom_old_node);

}

} // namespace FlexFlow
