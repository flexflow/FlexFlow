#include "test/utils/doctest.h"
#include "utils/graph/adjacency_digraph.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph.h"
#include "utils/graph/serialparallel.h"
#include "utils/graph/sp_ization.h"
#include <queue>
#include <tuple>
using namespace FlexFlow;

std::tuple<DiGraph, Node, Node> make_normal_nasnet_cell() {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> inputs = add_nodes(g, 2);
  std::vector<Node> sep = add_nodes(g, 5);
  std::vector<Node> id = add_nodes(g, 2);
  std::vector<Node> avg = add_nodes(g, 3);
  std::vector<Node> add = add_nodes(g, 5);
  std::vector<Node> concat = add_nodes(g, 1);

  std::vector<DirectedEdge> edges = {
      {inputs[0], sep[1]}, {inputs[0], id[1]},  {inputs[0], avg[1]},
      {inputs[0], avg[2]}, {inputs[0], sep[3]}, {inputs[0], sep[4]},
      {inputs[1], sep[0]}, {inputs[1], id[0]},  {inputs[1], avg[0]},
      {inputs[1], sep[2]}, {sep[0], add[0]},    {id[0], add[0]},
      {sep[1], add[1]},    {sep[2], add[1]},    {avg[0], add[2]},
      {id[1], add[2]},     {avg[1], add[3]},    {avg[2], add[3]},
      {sep[3], add[4]},    {sep[4], add[4]},
  };

  add_edges(g, edges);

  for (Node const &a : add) {
    g.add_edge({a, concat[0]});
  }

  return {g, inputs[0], inputs[1]};
}

std::tuple<DiGraph, Node, Node> make_reduction_nasnet_cell() {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> inputs = add_nodes(g, 2);
  std::vector<Node> sep = add_nodes(g, 5);
  std::vector<Node> id = add_nodes(g, 1);
  std::vector<Node> avg = add_nodes(g, 2);
  std::vector<Node> max = add_nodes(g, 2);
  std::vector<Node> add = add_nodes(g, 5);
  std::vector<Node> concat = add_nodes(g, 1);

  std::vector<DirectedEdge> edges = {
      {inputs[0], sep[0]}, {inputs[0], sep[2]}, {inputs[0], sep[3]},
      {inputs[1], max[1]}, {inputs[1], sep[1]}, {inputs[1], max[0]},
      {inputs[1], avg[0]}, {sep[0], add[0]},    {sep[1], add[0]},
      {max[0], add[1]},    {sep[2], add[1]},    {avg[0], add[2]},
      {sep[3], add[2]},    {max[1], add[3]},    {sep[4], add[3]},
      {avg[1], add[4]},    {id[0], add[4]},     {add[0], sep[4]},
      {add[0], avg[1]},    {add[1], id[0]},     {add[2], concat[0]},
      {add[3], concat[0]}, {add[4], concat[0]},
  };

  add_edges(g, edges);

  return {g, inputs[0], inputs[1]};
}

DiGraph make_cifar10(size_t num_reduction_cells, size_t N) {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  Node input = get_only(add_nodes(g, 1));
  std::deque<Node> outputting = {input, input, input};
  std::deque<Node> inputting;
  size_t num_cells = num_reduction_cells + N * (num_reduction_cells + 1);
  for (int i = 0; i < num_cells; i++) {
    auto [s, earlier_input, later_input] = (i % (N + 1) == N)
                                               ? make_reduction_nasnet_cell()
                                               : make_normal_nasnet_cell();
    // NOTE: the current flipped(s) in the next 3 lines are to account for the
    // DiEdge bug.
    assert(get_sources(flipped(s)).size() == 2);
    Node cell_output = get_only(get_sinks(flipped(s)));
    std::unordered_map<Node, Node> node_map = parallel_extend(g, flipped(s));
    later_input = node_map.at(later_input);
    earlier_input = node_map.at(earlier_input);
    cell_output = node_map.at(cell_output);

    outputting.push_back(cell_output);
    outputting.push_back(cell_output);
    inputting.push_back(earlier_input);
    inputting.push_back(later_input);

    Node a = outputting.front();
    Node b = inputting.front();
    inputting.pop_front();
    outputting.pop_front();
    g.add_edge({b, a});

    a = outputting.front();
    b = inputting.front();
    inputting.pop_front();
    outputting.pop_front();
    g.add_edge({b, a});

    assert(has_single_sink(g));
    assert(has_single_source(g));
    assert(is_acyclic(g));
    assert(outputting.size() == 3);
    assert(inputting.size() == 0);
  }
  return g;
}

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("Barrier Syncing SP-ization Algorithm") {
    /*
    digraph G {
    n0 [label="n0\nlayer=0"];
    n1 [label="n1\nlayer=1"];
    n2 [label="n2\nlayer=1"];
    n3 [label="n3\nlayer=2"];
    n4 [label="n4\nlayer=3"];
    n5 [label="n5\nlayer=4"];
    n6 [label="n6\nlayer=2"];

    n0 -> n1;
    n0 -> n2;
    n2 -> n3;
    n1 -> n4;
    n3 -> n4;
    n3 -> n5;
    n4 -> n5;
    n0 -> n6;
    n2 -> n6;
    n6 -> n5;
    }

    */

    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 7);

    std::vector<DirectedEdge> edges = {{n[0], n[1]},
                                       {n[0], n[2]},
                                       {n[2], n[3]},
                                       {n[1], n[4]},
                                       {n[3], n[4]},
                                       {n[3], n[5]},
                                       {n[4], n[5]},
                                       {n[0], n[6]},
                                       {n[2], n[6]},
                                       {n[6], n[5]}};

    add_edges(g, edges);

    auto gv = flipped(g); // to account for DiEdge bug
    SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
    CHECK(std::holds_alternative<Serial>(sp));

    Serial expected = Serial{{Parallel{{n[0]}},
                              Parallel{{n[1], n[2]}},
                              Parallel{{n[3], n[6]}},
                              Parallel{{n[4]}},
                              Parallel{{n[5]}}}};
    CHECK(std::get<Serial>(sp) == expected);
  }

  TEST_CASE("Dependency Invariant SP-ization algorithm - Straight Line") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 4);

    std::vector<DirectedEdge> edges = {
        {n[0], n[1]}, {n[1], n[2]}, {n[2], n[3]}};

    add_edges(g, edges);

    auto gv = flipped(g); // flipped to account for the DiEdge bug
    SerialParallelDecomposition result = dependency_invariant_sp_ization(gv);
    SerialParallelDecomposition expected = Serial{{n[0], n[1], n[2], n[3]}};
    CHECK(
        std::get<Serial>(result) ==
        std::get<Serial>(expected)); // currently cannot directly compare the 2.

    result = dependency_invariant_sp_ization_with_coalescing(gv);
    CHECK(
        std::get<Serial>(result) ==
        std::get<Serial>(expected)); // currently cannot directly compare the 2.
  }

  TEST_CASE("Dependency Invariant SP-ization algorithm - Rhombus Pattern") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 4);

    std::vector<DirectedEdge> edges = {
        {n[0], n[1]}, {n[0], n[2]}, {n[1], n[3]}, {n[2], n[3]}};

    add_edges(g, edges);

    auto gv = flipped(g); // flipped to account for the DiEdge bug
    SerialParallelDecomposition result = dependency_invariant_sp_ization(gv);
    SerialParallelDecomposition expected =
        Serial{{n[0], Parallel{{n[2], n[1]}}, n[3]}};

    result = dependency_invariant_sp_ization_with_coalescing(gv);
    CHECK(
        std::get<Serial>(result) ==
        std::get<Serial>(expected)); // currently cannot directly compare the 2.
  }

  TEST_CASE("Dependency Invariant SP-ization algorithm - Diamond Pattern") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 5);

    std::vector<DirectedEdge> edges = {{n[0], n[1]},
                                       {n[0], n[2]},
                                       {n[1], n[3]},
                                       {n[2], n[3]},
                                       {n[2], n[4]},
                                       {n[3], n[4]}};

    add_edges(g, edges);

    auto gv = flipped(g); // flipped to account for the DiEdge bug
    SUBCASE("Naive Version") {
      Serial result = std::get<Serial>(dependency_invariant_sp_ization(gv));
      Serial sp0 = {{n[0]}};
      Serial sp1 = {{n[0], n[1]}};
      Serial sp2 = {{n[0], n[2]}};
      Serial sp3 = {{Parallel{{sp2, sp1}}, n[3]}};
      Serial expected = {{Parallel{{sp3, sp2}}, n[4]}};
      CHECK(result == expected);
    }
    SUBCASE("Node coalescing") {
      Node s0 = n[0];
      Parallel p = {{Serial{{Parallel{{n[1], n[2]}}, n[3]}}, n[2]}};
      Node s1 = n[4];

      Serial expected = {{s0, p, s1}};

      Serial result =
          std::get<Serial>(dependency_invariant_sp_ization_with_coalescing(gv));
      CHECK(result == expected);
    }
  }

  TEST_CASE("Dependency Invariant SP-ization algorithm - More Complex Graph") {
    // Taken by "A New Algorithm for Mapping DAGs to Series-Parallel Form,
    // Escribano et Al, 2002"
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 18);

    std::vector<DirectedEdge> edges = {
        {n[0], n[1]},   {n[0], n[2]},   {n[1], n[3]},   {n[1], n[4]},
        {n[2], n[10]},  {n[2], n[11]},  {n[2], n[12]},  {n[3], n[5]},
        {n[3], n[6]},   {n[4], n[6]},   {n[4], n[7]},   {n[4], n[10]},
        {n[5], n[8]},   {n[6], n[8]},   {n[6], n[9]},   {n[7], n[8]},
        {n[8], n[17]},  {n[9], n[17]},  {n[10], n[16]}, {n[11], n[16]},
        {n[12], n[13]}, {n[12], n[14]}, {n[13], n[15]}, {n[14], n[15]},
        {n[15], n[16]}, {n[16], n[17]}};

    add_edges(g, edges);

    DiGraphView gv = flipped(g);
    SerialParallelDecomposition result =
        dependency_invariant_sp_ization_with_coalescing(gv);
    auto counter = node_counter(result);
    std::vector<size_t> expected_counts = {
        1, 2, 2, 3, 4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    for (size_t i = 0; i < n.size(); i++) {
      CHECK(counter[n[i]] == expected_counts[i]);
    }
    CHECK(node_count(result) == sum(expected_counts));
  }

  TEST_CASE(
      "Dependency Invariant SP-ization algorithm - Joining parallel segments") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> layer1 = add_nodes(g, 1);
    std::vector<Node> layer2 = add_nodes(g, 3);
    std::vector<Node> layer3 = add_nodes(g, 4);
    std::vector<Node> layer4 = add_nodes(g, 1);

    std::vector<DirectedEdge> edges;

    for (Node const &n1 : layer1) {
      for (Node const &n2 : layer2) {
        edges.push_back({n1, n2});
      }
    }

    for (Node const &n2 : layer2) {
      for (Node const &n3 : layer3) {
        edges.push_back({n2, n3});
      }
    }

    for (Node const &n3 : layer3) {
      for (Node const &n4 : layer4) {
        edges.push_back({n3, n4});
      }
    }

    add_edges(g, edges);

    DiGraphView gv = flipped(g); // flipped to account for the DiEdge bug

    Serial expected =
        Serial{{layer1[0],
                Parallel{{layer2[0], layer2[1], layer2[2]}},
                Parallel{{layer3[0], layer3[3], layer3[1], layer3[2]}},
                layer4[0]}};
    Serial result =
        std::get<Serial>(dependency_invariant_sp_ization_with_coalescing(gv));
    CHECK(result == expected);
  }

  TEST_CASE("Dependency Invariant SP-ization algorithm - NASNET-A like cell") {
    // From the TASO paper, pg 57
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    Node root = get_only(add_nodes(g, 1));
    std::vector<Node> input = add_nodes(g, 2);
    std::vector<Node> dwc = add_nodes(g, 5);
    std::vector<Node> conv = add_nodes(g, 5);
    std::vector<Node> avg = add_nodes(g, 3);
    std::vector<Node> add = add_nodes(g, 5);
    Node concat = get_only(add_nodes(g, 1));

    std::vector<DirectedEdge> edges = {
        {root, input[0]},   {root, input[1]},   {input[0], dwc[0]},
        {input[0], dwc[1]}, {input[0], avg[0]}, {input[0], avg[1]},
        {input[0], avg[2]}, {input[0], dwc[2]}, {input[1], add[2]},
        {input[1], dwc[3]}, {input[1], dwc[4]}, {input[1], add[4]},
        {dwc[0], conv[0]},  {dwc[1], conv[1]},  {dwc[2], conv[2]},
        {dwc[3], conv[3]},  {dwc[4], conv[4]},  {conv[0], add[0]},
        {conv[1], add[0]},  {avg[0], add[1]},   {avg[1], add[1]},
        {avg[2], add[2]},   {conv[2], add[3]},  {conv[3], add[3]},
        {conv[4], add[4]}};

    add_edges(g, edges);

    for (auto const &a : add) {
      g.add_edge({a, concat});
    }

    SUBCASE("No coalescing") {
      DiGraphView gv = flipped(g);
      SerialParallelDecomposition result = dependency_invariant_sp_ization(gv);

      int extranodes =
          get_nodes(multidigraph_from_sp_decomposition(result)).size() -
          get_nodes(gv).size();
      CHECK(extranodes == 17);
    }
    SUBCASE("coalescing") {
      DiGraphView gv = flipped(g);
      SerialParallelDecomposition result =
          dependency_invariant_sp_ization_with_coalescing(gv);

      std::unordered_set<Node> nodes_in_graph = get_nodes(g);
      std::unordered_set<Node> nodes_in_sp = get_nodes(result);
      CHECK(nodes_in_graph == nodes_in_sp);
      size_t extranodes = 4;
      CHECK(node_count(result) == (nodes_in_graph.size() + extranodes));
    }
  }

  TEST_CASE("Dependency invariant SP-graph - NASNET like structure") {
    DiGraph g = make_cifar10(1, 1);
    DiGraphView gv = flipped(g);
    SerialParallelDecomposition result =
        dependency_invariant_sp_ization_with_coalescing(gv);
    // CHECK(get_nodes(g).size() == node_count(result));
  }
}
