#include <doctest/doctest.h>
#include "utils/graph/serial_parallel/parallel_reduction.h"
#include "utils/graph/serial_parallel/serialparallel_internal.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/instances/adjacency_multidigraph.h"
#include "utils/graph/algorithms.h"
#include "utils/fmt/variant.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/multidigraph/algorithms/add_nodes.h"
#include "utils/graph/multidigraph/algorithms/add_edges.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  // TEST_CASE("find_bottleneck_node") {
  //
  // }

  TEST_CASE("from_source_to_sink") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    std::vector<Node> n = add_nodes(g, 7);
    
    add_edges(g, {
      DirectedEdge{n.at(0), n.at(1)},
      DirectedEdge{n.at(0), n.at(2)},
      DirectedEdge{n.at(1), n.at(3)},
      DirectedEdge{n.at(2), n.at(3)},
      DirectedEdge{n.at(3), n.at(4)},
      DirectedEdge{n.at(3), n.at(5)},
      DirectedEdge{n.at(4), n.at(6)},
      DirectedEdge{n.at(5), n.at(6)},
    });
    
    SUBCASE("without settings") {
      std::unordered_set<Node> result = from_source_to_sink(g, {n.at(0)}, {n.at(3)});
      std::unordered_set<Node> correct = { n.at(0), n.at(1), n.at(2), n.at(3) };

      CHECK(result == correct);
    }

    auto get_result_for_settings = [&](SourceSettings const &source_settings,
                                       SinkSettings const &sink_settings) {
      return from_source_to_sink(g, {n.at(0)}, {n.at(3)}, source_settings, sink_settings);
    };

    SUBCASE("include src and sink") {
      std::unordered_set<Node> result = get_result_for_settings(SourceSettings::INCLUDE_SOURCE_NODES, SinkSettings::INCLUDE_SINK_NODES);
      std::unordered_set<Node> correct = { n.at(0), n.at(1), n.at(2), n.at(3) };

      CHECK(result == correct);
    }
    
    SUBCASE("include src, exclude sink") {
      std::unordered_set<Node> result = get_result_for_settings(SourceSettings::INCLUDE_SOURCE_NODES, SinkSettings::EXCLUDE_SINK_NODES);
      std::unordered_set<Node> correct = { n.at(0), n.at(1), n.at(2) };

      CHECK(result == correct);
    }

    SUBCASE("exclude src, include sink") {
      std::unordered_set<Node> result = get_result_for_settings(SourceSettings::EXCLUDE_SOURCE_NODES, SinkSettings::INCLUDE_SINK_NODES);
      std::unordered_set<Node> correct = { n.at(1), n.at(2), n.at(3)};

      CHECK(result == correct);
    }

    SUBCASE("exclude src, exclude sink") {
      std::unordered_set<Node> result = get_result_for_settings(SourceSettings::EXCLUDE_SOURCE_NODES, SinkSettings::EXCLUDE_SINK_NODES);
      std::unordered_set<Node> correct = {n.at(1), n.at(2)};

      CHECK(result == correct);
    }
  }

  TEST_CASE("find_bottleneck_node") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();

    SUBCASE("has bottleneck") {
      std::vector<Node> n = add_nodes(g, 2);
      g.add_edge(DirectedEdge{n.at(0), n.at(1)});

      std::optional<Node> result = find_bottleneck_node(g);
      std::optional<Node> correct = n.at(1);

      CHECK(result == correct);
    }

    SUBCASE("is parallel") {
      std::vector<Node> n = add_nodes(g, 2);

      std::optional<Node> result = find_bottleneck_node(g);
      std::optional<Node> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("is only a single node") {
      std::vector<Node> n = add_nodes(g, 1);

      std::optional<Node> result = find_bottleneck_node(g);
      std::optional<Node> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
  
  TEST_CASE("sp_decomposition (base case)") { 
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    Node n = g.add_node();
    std::optional<std::variant<IntermediateSpDecompositionTree, Node>> result = sp_decomposition(g);
    std::optional<std::variant<IntermediateSpDecompositionTree, Node>> correct = n;
    CHECK(result == correct);
  }

  // TEST_CASE("sp_decomposition (parallel)") { 
  //   DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  //   std::vector<Node> ns = add_nodes(g, 2);
  //   std::optional<std::variant<IntermediateSpDecompositionTree, Node>> result = sp_decomposition(g);
  //   std::optional<std::variant<IntermediateSpDecompositionTree, Node>> correct = IntermediateSpDecompositionTree{
  //     SplitType::PARALLEL,
  //     {ns.at(0), ns.at(1)},
  //   };
  //   CHECK(result == correct);
  // }

  TEST_CASE("sp_decomposition (serial)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> ns = add_nodes(g, 2);
    g.add_edge(DirectedEdge{ns.at(0), ns.at(1)});
    std::optional<std::variant<IntermediateSpDecompositionTree, Node>> result = sp_decomposition(g);
    std::optional<std::variant<IntermediateSpDecompositionTree, Node>> correct = IntermediateSpDecompositionTree{
      SplitType::SERIAL,
      {ns.at(0), ns.at(1)},
    };
    CHECK(result == correct);
  }

  TEST_CASE("sp_decomposition (composite)") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> ns = add_nodes(g, 3);
    add_edges(g, {
      DirectedEdge{ns.at(0), ns.at(1)},
      DirectedEdge{ns.at(0), ns.at(2)},
    });
    std::optional<std::variant<IntermediateSpDecompositionTree, Node>> result = sp_decomposition(g);
    std::optional<std::variant<IntermediateSpDecompositionTree, Node>> correct = IntermediateSpDecompositionTree{
      SplitType::SERIAL,
      {
        ns.at(0),
        IntermediateSpDecompositionTree{
          SplitType::PARALLEL, 
          {
            ns.at(1),
            ns.at(2),
          }
        }
      }
    };
    CHECK(result == correct);
  }

  // TEST_CASE("sp_decomposition (hmm)") {
  //   DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  //
  //   std::vector<Node> n = add_nodes(g, 6);
  //
  //   add_edges(g, {
  //     DirectedEdge{n.at(0), n.at(1)},
  //     DirectedEdge{n.at(0), n.at(2)},
  //     DirectedEdge{n.at(1), n.at(3)},
  //     DirectedEdge{n.at(2), n.at(4)},
  //     DirectedEdge{n.at(3), n.at(5)},
  //     DirectedEdge{n.at(4), n.at(5)},
  //   });
  // }
  // TEST_CASE("parallel_decomposition") {
  //   DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  //   std::vector<Node> ns = add_nodes(g, 5);
  //   add_edges(g, {
  //     DirectedEdge{ns.at(0), ns.at(2)},
  //     DirectedEdge{ns.at(1), ns.at(3)},
  //     DirectedEdge{ns.at(1), ns.at(4)},
  //     DirectedEdge{ns.at(3), ns.at(4)},
  //   });
  //
  //   IntermediateSpDecompositionTree result = parallel_decomposition(g);
  //   IntermediateSpDecompositionTree correct = IntermediateSpDecompositionTree{
  //     SplitType::PARALLEL,
  //
  // }
}
