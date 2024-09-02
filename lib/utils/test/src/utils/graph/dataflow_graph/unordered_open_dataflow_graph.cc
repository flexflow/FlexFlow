#include <doctest/doctest.h>
#include "utils/graph/dataflow_graph/dataflow_edge_query.h"
#include "utils/graph/dataflow_graph/dataflow_graph.h"
#include "utils/graph/dataflow_graph/dataflow_output_query.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/node/node_query.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("UnorderedSetDataflowGraph") {
    DataflowGraph g = DataflowGraph::create<UnorderedSetDataflowGraph>();

    {
      std::unordered_set<Node> result = g.query_nodes(node_query_all());
      std::unordered_set<Node> correct = {};
      REQUIRE(result == correct);
    }

    {
      std::unordered_set<DataflowEdge> result =
          g.query_edges(dataflow_edge_query_all());
      std::unordered_set<DataflowEdge> correct = {};
      REQUIRE(result == correct);
    }

    {
      std::unordered_set<DataflowOutput> result =
          g.query_outputs(dataflow_output_query_all());
      std::unordered_set<DataflowOutput> correct = {};
      REQUIRE(result == correct);
    }

    NodeAddedResult added = g.add_node({}, 2);

    {
      std::unordered_set<Node> result = g.query_nodes(node_query_all());
      std::unordered_set<Node> correct = {added.node};
      REQUIRE(result == correct);
    }

    {
      std::unordered_set<DataflowEdge> result =
          g.query_edges(dataflow_edge_query_all());
      std::unordered_set<DataflowEdge> correct = {};
      REQUIRE(result == correct);
    }

    {
      std::unordered_set<DataflowOutput> result =
          g.query_outputs(dataflow_output_query_all());
      std::unordered_set<DataflowOutput> correct =
          unordered_set_of(added.outputs);
      REQUIRE(result == correct);
    }

    NodeAddedResult added2 = g.add_node(added.outputs, 3);

    {
      std::unordered_set<Node> result = g.query_nodes(node_query_all());
      std::unordered_set<Node> correct = {added.node, added2.node};
      REQUIRE(result == correct);
    }

    {
      std::unordered_set<DataflowEdge> result =
          g.query_edges(dataflow_edge_query_all());
      std::unordered_set<DataflowEdge> correct = {
          DataflowEdge{added.outputs.at(0), DataflowInput{added2.node, 0}},
          DataflowEdge{added.outputs.at(1), DataflowInput{added2.node, 1}},
      };
      REQUIRE(result == correct);
    }

    {
      std::unordered_set<DataflowOutput> result =
          g.query_outputs(dataflow_output_query_all());
      std::unordered_set<DataflowOutput> correct = set_union(
          unordered_set_of(added.outputs), unordered_set_of(added2.outputs));
      REQUIRE(result == correct);
    }
  }
}
