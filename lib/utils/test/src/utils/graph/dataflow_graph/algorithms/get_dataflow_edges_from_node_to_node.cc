#include "utils/graph/dataflow_graph/algorithms/get_dataflow_edges_from_node_to_node.h"
#include <doctest/doctest.h>
#include "utils/containers/get_only.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/dataflow_graph/dataflow_graph.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_dataflow_edges_from_node_to_node") {
    DataflowGraph g = DataflowGraph::create<UnorderedSetDataflowGraph>();

    SUBCASE("gets edges if there are multiple") {
      NodeAddedResult n1_added = g.add_node({}, 2);
      Node n1 = n1_added.node;
      DataflowOutput n1_o0 = n1_added.outputs.at(0);
      DataflowOutput n1_o1 = n1_added.outputs.at(1);

      NodeAddedResult n2_added = g.add_node({n1_o0, n1_o0, n1_o1}, 0);
      Node n2 = n2_added.node;

      std::unordered_set<DataflowEdge> result = get_dataflow_edges_from_node_to_node(g, n1, n2);
      std::unordered_set<DataflowEdge> correct = {
        DataflowEdge{
          n1_o0,
          DataflowInput{n2, 0},
        },
        DataflowEdge{
          n1_o0,
          DataflowInput{n2, 1},
        },
        DataflowEdge{
          n1_o1,
          DataflowInput{n2, 2},
        },
      };

      CHECK(result == correct);
    }

    SUBCASE("does not get edges to/from other nodes") {
      NodeAddedResult n1_added = g.add_node({}, 1);
      Node n1 = n1_added.node;
      DataflowOutput o1 = get_only(n1_added.outputs);

      NodeAddedResult n2_added = g.add_node({o1}, 1);
      Node n2 = n2_added.node;
      DataflowOutput o2 = get_only(n2_added.outputs);

      NodeAddedResult n3_added = g.add_node({o2}, 1);
      Node n3 = n3_added.node;
      DataflowOutput o3 = get_only(n3_added.outputs);

      std::unordered_set<DataflowEdge> result = get_dataflow_edges_from_node_to_node(g, n1, n3);
      std::unordered_set<DataflowEdge> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("does not get flipped edges (i.e., respects from vs to direction)") {
      NodeAddedResult n1_added = g.add_node({}, 1);
      Node n1 = n1_added.node;
      DataflowOutput o1 = get_only(n1_added.outputs);

      NodeAddedResult n2_added = g.add_node({o1}, 0);
      Node n2 = n2_added.node;

      std::unordered_set<DataflowEdge> result = get_dataflow_edges_from_node_to_node(g, n2, n1);
      std::unordered_set<DataflowEdge> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("returns empty set if no edges exist between the given nodes") {
      NodeAddedResult n1_added = g.add_node({}, 1);
      Node n1 = n1_added.node;

      NodeAddedResult n2_added = g.add_node({}, 1);
      Node n2 = n2_added.node;

      std::unordered_set<DataflowEdge> result = get_dataflow_edges_from_node_to_node(g, n1, n2);
      std::unordered_set<DataflowEdge> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("returns empty set if src node == dst node (as cycles cannot exist in DataflowGraph") {
      NodeAddedResult n1_added = g.add_node({}, 1);
      Node n1 = n1_added.node;

      std::unordered_set<DataflowEdge> result = get_dataflow_edges_from_node_to_node(g, n1, n1);
      std::unordered_set<DataflowEdge> correct = {};

      CHECK(result == correct);
    }
  }
}
