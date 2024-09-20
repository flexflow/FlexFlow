#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/containers/get_only.h"
#include "utils/fmt/unordered_set.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_inputs/get_outputs") {
    DataflowGraph g = DataflowGraph::create<UnorderedSetDataflowGraph>();

    NodeAddedResult n1_added = g.add_node({}, 1);
    Node n1 = n1_added.node;
    DataflowOutput o1 = get_only(n1_added.outputs);

    NodeAddedResult n2_added = g.add_node({}, 1);
    Node n2 = n2_added.node;
    DataflowOutput o2 = get_only(n2_added.outputs);

    NodeAddedResult n3_added = g.add_node({}, 1);
    Node n3 = n3_added.node;
    DataflowOutput o3 = get_only(n3_added.outputs);

    NodeAddedResult n4_added = g.add_node({o1, o2, o3}, 1);
    Node n4 = n4_added.node;
    DataflowOutput o4 = get_only(n4_added.outputs);

    SUBCASE("get_input_values") {
      std::vector<DataflowOutput> result = get_input_values(g, n4);
      std::vector<DataflowOutput> correct = {o1, o2, o3};
      CHECK(result == correct);
    }

    SUBCASE("get_outputs") {
      std::vector<DataflowOutput> result = get_outputs(g, n4);
      std::vector<DataflowOutput> correct = {o4};
      CHECK(result == correct);
    }
  }

  TEST_CASE("topological_ordering") {
    DataflowGraph g = DataflowGraph::create<UnorderedSetDataflowGraph>();

    NodeAddedResult n1_added = g.add_node({}, 1);
    Node n1 = n1_added.node;
    DataflowOutput o1 = get_only(n1_added.outputs);

    NodeAddedResult n2_added = g.add_node({o1}, 1);
    Node n2 = n2_added.node;
    DataflowOutput o2 = get_only(n2_added.outputs);

    NodeAddedResult n3_added = g.add_node({o2}, 1);
    Node n3 = n3_added.node;
    DataflowOutput o3 = get_only(n3_added.outputs);

    std::vector<Node> result = get_topological_ordering(g);
    std::vector<Node> correct = {n1, n2, n3};
    CHECK(result == correct);
  }
}
