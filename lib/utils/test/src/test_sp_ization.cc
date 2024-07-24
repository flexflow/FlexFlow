#include "test/utils/doctest.h"
#include "utils/graph/adjacency_digraph.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph.h"
#include "utils/graph/serialparallel.h"
#include "utils/graph/sp_ization.h"
#include <queue>
#include <tuple>
using namespace FlexFlow;

bool isclose(float a, float b, float rel_tol = 1e-5f) {
  return std::fabs(a - b) <= rel_tol * std::fmax(std::fabs(a), std::fabs(b));
}

namespace Distributions {
struct Constant {
  float val;
  Constant(float val = 1) : val(val) {}
  float operator()() const {
    return val;
  }
};

struct Uniform {
  float a, b;
  Uniform(float a = 0, float b = 1) : a(a), b(b) {}
  float operator()() const {
    return a + ((static_cast<double>(std::rand()) / RAND_MAX) * (b - a));
  }
};

struct Bernoulli {
  float p;
  Bernoulli(float p = 0.5) : p(p) {}
  float operator()() const {
    return (Uniform(0, 1)() < p);
  }
};

struct Binary {
  float a, b, p;
  Binary(float a = 0, float b = 1, float p = 0.5) : a(a), b(b), p(p) {}
  float operator()() const {
    return (Bernoulli(p)() ? a : b);
  }
};

template <typename Dist>
std::unordered_map<Node, float>
    make_cost_map(std::unordered_set<Node> const &nodes,
                  Dist const &distribution) {
  std::unordered_map<Node, float> cost_map;
  for (Node const &node : nodes) {
    cost_map[node] = distribution();
  }
  return cost_map;
}
} // namespace Distributions

namespace TestingGraphs {

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

DiGraph make_linear(size_t length) {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  if (length == 0) {
    return g;
  }
  std::vector<Node> nodes = add_nodes(g, length);

  for (size_t i = 0; i < length - 1; ++i) {
    g.add_edge({nodes[i], nodes[i + 1]});
  }

  return g;
}

DiGraph make_rhombus() {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> n = add_nodes(g, 4);

  std::vector<DirectedEdge> edges = {
      {n[0], n[1]}, {n[0], n[2]}, {n[1], n[3]}, {n[2], n[3]}};

  add_edges(g, edges);
  return g;
}

DiGraph make_diamond() {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> n = add_nodes(g, 6);

  std::vector<DirectedEdge> edges = {
      {n[0], n[1]},
      {n[0], n[2]},
      {n[1], n[3]},
      {n[2], n[3]},
      {n[2], n[4]},
      {n[3], n[5]},
      {n[4], n[5]},
  };

  add_edges(g, edges);
  return g;
}

DiGraph make_fully_connected(std::vector<size_t> layer_sizes) {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<std::vector<Node>> layers =
      transform(layer_sizes, [&g](size_t size) { return add_nodes(g, size); });

  std::vector<DirectedEdge> edges;

  for (size_t i = 0; i < layers.size() - 1; ++i) {
    for (Node const &n1 : layers[i]) {
      for (Node const &n2 : layers[i + 1]) {
        edges.push_back({n1, n2});
      }
    }
  }

  add_edges(g, edges);
  return g;
}

DiGraph make_parallel_chains(size_t chain_length, size_t chain_num) {
  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  assert(chain_length >= 3);
  assert(chain_num >= 1);
  std::vector<std::vector<Node>> chains;

  for (size_t i = 0; i < chain_num; i++) {
    std::vector<Node> chain_nodes = add_nodes(g, chain_length - 2);
    chains.push_back(chain_nodes);

    for (size_t j = 0; j < chain_length - 3; j++) {
      g.add_edge({chain_nodes[j], chain_nodes[j + 1]});
    }
  }

  Node source = get_only(add_nodes(g, 1));
  Node sink = get_only(add_nodes(g, 1));

  for (auto const &chain : chains) {
    g.add_edge({source, chain.front()});
    g.add_edge({chain.back(), sink});
  }

  return g;
}

DiGraph make_sample_dag_1() {

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
  return g;
}

DiGraph make_sample_dag_2() {
  NOT_IMPLEMENTED();
}

DiGraph make_sample_dag_3() {
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
  return g;
}

DiGraph make_taso_nasnet_cell() {
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
  return g;
}

} // namespace TestingGraphs

namespace BenchMark {
using Result = std::pair<float, float>;
using CombinedResult = std::tuple<Result, Result, Result>;
template <typename D>
CombinedResult perform_benchmark(DiGraphView const &g,
                                 D const &Dist,
                                 size_t const &repeat) {
  Result dependency_invariant = {0, 0};
  Result barrier_sync = {0, 0};
  Result cost_aware = {0, 0};
  for (int i = 0; i < repeat; i++) {
    auto cost_map = Distributions::make_cost_map(get_nodes(g), Dist);
    SerialParallelDecomposition sp1 =
        dependency_invariant_sp_ization_with_coalescing(g);
    SerialParallelDecomposition sp2 = barrier_sync_sp_ization(g);
    SerialParallelDecomposition sp3 =
        cost_aware_barrier_sync_sp_ization(g, cost_map);

    dependency_invariant.first += relative_cost_increase(g, sp1, cost_map);
    dependency_invariant.second +=
        relative_critical_path_cost_increase(g, sp1, cost_map);
    barrier_sync.first += relative_cost_increase(g, sp2, cost_map);
    barrier_sync.second +=
        relative_critical_path_cost_increase(g, sp2, cost_map);
    cost_aware.first += relative_cost_increase(g, sp3, cost_map);
    cost_aware.second += relative_critical_path_cost_increase(g, sp3, cost_map);
  }
  std::vector<Result> results = {
      dependency_invariant, barrier_sync, cost_aware};
  for (Result &r : results) {
    r.first /= repeat;
    r.second /= repeat;
  }

  return {results[0], results[1], results[2]};
}

void output_benchmark(CombinedResult const &combined_result,
                      std::string const &title) {
  auto [d_i, b_s, c_a] = combined_result;
  std::cout << "Benchmark for " << title << std::endl;
  std::cout << "Technique | Work-Increase | Critical-Path-Increase"
            << std::endl;
  std::cout << "Dependency Invariant | " << d_i.first << " | " << d_i.second
            << std::endl;
  std::cout << "Barrier Sync | " << b_s.first << " | " << b_s.second
            << std::endl;
  std::cout << "Cost_Increase | " << c_a.first << " | " << c_a.second
            << std::endl;
  std::cout << std::endl;
}

template <typename D>
void bench_mark(std::string title,
                DiGraphView const &g,
                D const &Dist,
                size_t const &repeat) {
  output_benchmark(perform_benchmark(g, Dist, repeat), title);
}

} // namespace BenchMark

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("barrier_sync_sp_ization - sample_dag_1") {

    DiGraph g = TestingGraphs::make_sample_dag_1();
    std::vector<Node> n = sorted(get_nodes(g));
    auto gv = flipped(g); // to account for DiEdge bug
    SerialParallelDecomposition result = barrier_sync_sp_ization(gv);
    Serial expected = Serial{
        {n[0], Parallel{{n[1], n[2]}}, Parallel{{n[3], n[6]}}, n[4], n[5]}};
    CHECK(std::get<Serial>(result) == expected);
  }

  TEST_CASE("dependency_invariant_sp_ization - linear") {
    DiGraph g = TestingGraphs::make_linear(4);
    std::vector<Node> n = sorted(get_nodes(g));
    auto gv = flipped(g); // flipped to account for the DiEdge bug
    SerialParallelDecomposition result = dependency_invariant_sp_ization(gv);
    SerialParallelDecomposition expected = Serial{{n[0], n[1], n[2], n[3]}};
    CHECK(
        std::get<Serial>(result) ==
        std::get<Serial>(expected)); // currently cannot directly compare the 2.
  }

  TEST_CASE("dependency_invariant_sp_ization - rhombus") {
    DiGraph g = TestingGraphs::make_rhombus();
    std::vector<Node> n = sorted(get_nodes(g));

    auto gv = flipped(g); // flipped to account for the DiEdge bug
    SerialParallelDecomposition result = dependency_invariant_sp_ization(gv);
    SerialParallelDecomposition expected =
        Serial{{Parallel{{Serial{{n[0], n[1]}}, Serial{{n[0], n[2]}}}}, n[3]}};
    CHECK(std::get<Serial>(result) == std::get<Serial>(expected));
  }

  TEST_CASE("dependency_invariant_sp_ization - diamond") {
    DiGraph g = TestingGraphs::make_diamond();
    std::vector<Node> n = sorted(get_nodes(g));

    auto gv = flipped(g); // flipped to account for the DiEdge bug
    Serial result = std::get<Serial>(dependency_invariant_sp_ization(gv));
    Serial sp0 = {{n[0]}};
    Serial sp1 = {{n[0], n[1]}};
    Serial sp2 = {{n[0], n[2]}};
    Serial sp3 = {{Parallel{{sp2, sp1}}, n[3]}};
    Serial sp4 = {{n[0], n[2], n[4]}};
    Serial expected = {{Parallel{{sp3, sp4}}, n[5]}};
    CHECK(result == expected);
  }

  TEST_CASE("dependency_invariant_sp_ization_with - taso_nasnet_cell") {

    DiGraph g = TestingGraphs::make_taso_nasnet_cell();

    DiGraphView gv = flipped(g);
    SerialParallelDecomposition result = dependency_invariant_sp_ization(gv);

    int extranodes =
        get_nodes(multidigraph_from_sp_decomposition(result)).size() -
        get_nodes(gv).size();
    CHECK(extranodes == 17);
  }

  TEST_CASE("dependency_invariant_sp_ization_with_coalescing - linear") {
    DiGraph g = TestingGraphs::make_linear(4);
    std::vector<Node> n = sorted(get_nodes(g));

    auto gv = flipped(g); // flipped to account for the DiEdge bug
    SerialParallelDecomposition result =
        dependency_invariant_sp_ization_with_coalescing(gv);
    SerialParallelDecomposition expected = Serial{{n[0], n[1], n[2], n[3]}};
    CHECK(
        std::get<Serial>(result) ==
        std::get<Serial>(expected)); // currently cannot directly compare the 2.
  }

  TEST_CASE("dependency_invariant_sp_ization_with_coalescing - rhombus") {
    DiGraph g = TestingGraphs::make_rhombus();
    std::vector<Node> n = sorted(get_nodes(g));

    auto gv = flipped(g); // flipped to account for the DiEdge bug
    SerialParallelDecomposition result =
        dependency_invariant_sp_ization_with_coalescing(gv);
    SerialParallelDecomposition expected =
        Serial{{n[0], Parallel{{n[2], n[1]}}, n[3]}};

    CHECK(
        std::get<Serial>(result) ==
        std::get<Serial>(expected)); // currently cannot directly compare the 2.
  }

  TEST_CASE("dependency_invariant_sp_ization_with_coalescing - diamond") {
    DiGraph g = TestingGraphs::make_diamond();
    std::vector<Node> n = sorted(get_nodes(g));

    auto gv = flipped(g); // flipped to account for the DiEdge bug
    Node s0 = n[0];
    Parallel p = {
        {Serial{{Parallel{{n[1], n[2]}}, n[3]}}, Serial{{n[2], n[4]}}}};
    Node s1 = n[5];

    Serial expected = {{s0, p, s1}};

    Serial result =
        std::get<Serial>(dependency_invariant_sp_ization_with_coalescing(gv));
    CHECK(result == expected);
  }

  TEST_CASE("dependency_invariant_sp_ization_with_coalescing - sample_dag_3") {
    DiGraph g = TestingGraphs::make_sample_dag_3();

    DiGraphView gv = flipped(g);
    SerialParallelDecomposition result =
        dependency_invariant_sp_ization_with_coalescing(gv);
    auto counter = node_counter(result);
    std::vector<size_t> expected_counts = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4};
    std::vector<size_t> result_counts = sorted(values(counter));
    CHECK(expected_counts == result_counts);
    CHECK(node_count(result) == sum(expected_counts));
  }

  TEST_CASE(
      "dependency_invariant_sp_ization_with_coalescing - fully_connected") {
    DiGraph g = TestingGraphs::make_fully_connected({1, 4, 6, 1});
    std::vector<Node> n = sorted(get_nodes(g));

    DiGraphView gv = flipped(g); // flipped to account for the DiEdge bug

    Serial expected = Serial{{n[0],
                              Parallel{{n[1], n[2], n[3], n[4]}},
                              Parallel{{n[5], n[6], n[7], n[8], n[9], n[10]}},
                              n[11]}};
    Serial result =
        std::get<Serial>(dependency_invariant_sp_ization_with_coalescing(gv));
    CHECK(result == expected);
  }

  TEST_CASE(
      "dependency_invariant_sp_ization_with_coalescing - taso_nasnet_cell") {

    DiGraph g = TestingGraphs::make_taso_nasnet_cell();

    DiGraphView gv = flipped(g);
    SerialParallelDecomposition result =
        dependency_invariant_sp_ization_with_coalescing(gv);

    std::unordered_set<Node> nodes_in_graph = get_nodes(g);
    std::unordered_set<Node> nodes_in_sp = get_nodes(result);
    CHECK(nodes_in_graph == nodes_in_sp);
    size_t extranodes = 4;
    CHECK(node_count(result) == (nodes_in_graph.size() + extranodes));
  }

  TEST_CASE("dependency_invariant_sp_ization_with_coalescing - Cifar10") {
    DiGraph g = TestingGraphs::make_cifar10(1, 1);
    DiGraphView gv = flipped(g);
    SerialParallelDecomposition result =
        dependency_invariant_sp_ization_with_coalescing(gv);
    std::unordered_set<Node> nodes_in_graph = get_nodes(gv);
    std::unordered_set<Node> nodes_in_sp = get_nodes(result);
    CHECK(nodes_in_graph == nodes_in_sp);
    CHECK(node_count(result) >= node_count(gv));
  }

  TEST_CASE("cost_aware_barrier_sync_sp_ization - linear") {
    DiGraph g = TestingGraphs::make_linear(4);
    DiGraphView gv = flipped(g);
    std::vector<Node> n = sorted(get_nodes(gv));
    auto cost_map =
        Distributions::make_cost_map(get_nodes(g), Distributions::Constant(1));
    SerialParallelDecomposition result =
        cost_aware_barrier_sync_sp_ization(gv, cost_map);
    CHECK(std::get<Serial>(result) == Serial{{n[0], n[1], n[2], n[3]}});
  }

  TEST_CASE("cost_aware_barrier_sync_sp_ization - rhombus") {
    DiGraph g = TestingGraphs::make_rhombus();
    DiGraphView gv = flipped(g);
    std::vector<Node> n = sorted(get_nodes(g));
    auto cost_map =
        Distributions::make_cost_map(get_nodes(g), Distributions::Constant(1));
    SerialParallelDecomposition result =
        cost_aware_barrier_sync_sp_ization(gv, cost_map);
    CHECK(std::get<Serial>(result) ==
          Serial{{n[0], Parallel{{n[1], n[2]}}, n[3]}});
  }

  TEST_CASE("cost_aware_barrier_sync_sp_ization - custom graph") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    std::vector<Node> n = add_nodes(g, 5);
    std::vector<DirectedEdge> edges = {
        {n[0], n[1]}, {n[0], n[2]}, {n[1], n[3]}, {n[2], n[4]}, {n[3], n[4]}};
    add_edges(g, edges);
    std::unordered_map<Node, float> cost_map = {
        {n[0], 1}, {n[1], 4}, {n[2], 10}, {n[3], 4}, {n[4], 1}};
    DiGraphView gv = flipped(g);

    SerialParallelDecomposition result =
        cost_aware_barrier_sync_sp_ization(gv, cost_map);
    Serial expected =
        Serial{{n[0], Parallel{{Serial{{n[1], n[3]}}, n[2]}}, n[4]}};
    CHECK(std::get<Serial>(result) == expected);
  }

  TEST_CASE("cost_aware_barrier_sync_sp_ization - sample_dag_1") {
    DiGraph g = TestingGraphs::make_sample_dag_1();
    std::vector<Node> n = sorted(get_nodes(g));
    DiGraphView gv = flipped(g);

    SUBCASE("Cost Map #1") {
      std::unordered_map<Node, float> cost_map = {{n[0], 1},
                                                  {n[1], 5},
                                                  {n[2], 3},
                                                  {n[3], 1},
                                                  {n[4], 1},
                                                  {n[5], 1},
                                                  {n[6], 1}};
      SerialParallelDecomposition result =
          cost_aware_barrier_sync_sp_ization(gv, cost_map);
      Serial expected =
          Serial{{n[0],
                  Parallel{{n[1], Serial{{n[2], Parallel{{n[3], n[6]}}}}}},
                  n[4],
                  n[5]}};
      CHECK(std::get<Serial>(result) == expected);
    }
    SUBCASE("Cost Map #2") {
      std::unordered_map<Node, float> cost_map = {{n[0], 1},
                                                  {n[1], 5},
                                                  {n[2], 3},
                                                  {n[3], 2},
                                                  {n[4], 4},
                                                  {n[5], 1},
                                                  {n[6], 2}};
      SerialParallelDecomposition result =
          cost_aware_barrier_sync_sp_ization(gv, cost_map);
      Serial expected = Serial{
          {n[0], Parallel{{n[1], Serial{{n[2], n[6]}}}}, n[3], n[4], n[5]}};
      CHECK(std::get<Serial>(result) == expected);
    }
  }

  TEST_CASE("barrier_sync_sp_ization - cost increases") {
    SUBCASE("linear") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_linear(10);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_linear(10);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_linear(10);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("fully_connected") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_fully_connected({1, 4, 6, 4, 1});
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_fully_connected({1, 4, 6, 4, 1});
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_fully_connected({1, 4, 6, 4, 1});
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("sample_dag_3") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_sample_dag_3();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(relative_critical_path_cost_increase(gv, sp, cost_map) == 1);
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_sample_dag_3();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_sample_dag_3();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("sample_dag_1") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_sample_dag_1();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_sample_dag_1();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_sample_dag_1();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("taso_nasnet_cell") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_taso_nasnet_cell();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_taso_nasnet_cell();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_taso_nasnet_cell();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("parallel_chains") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_parallel_chains(20, 4);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_parallel_chains(20, 4);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_parallel_chains(20, 4);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("cifar10") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_cifar10(2, 2);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_cifar10(2, 2);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_cifar10(2, 2);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp = barrier_sync_sp_ization(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
    }
  }

  TEST_CASE("cost_aware_barrier_sync_sp_ization - cost increases") {
    SUBCASE("linear") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_linear(10);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_linear(10);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_linear(10);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("fully_connected") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_fully_connected({1, 4, 6, 4, 1});
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_fully_connected({1, 4, 6, 4, 1});
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_fully_connected({1, 4, 6, 4, 1});
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("sample_dag_3") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_sample_dag_3();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(relative_critical_path_cost_increase(gv, sp, cost_map) == 1);
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_sample_dag_3();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_sample_dag_3();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("sample_dag_1") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_sample_dag_1();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_sample_dag_1();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_sample_dag_1();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("taso_nasnet_cell") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_taso_nasnet_cell();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_taso_nasnet_cell();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_taso_nasnet_cell();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("cifar10") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_cifar10(2, 1);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_cifar10(2, 1);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_cifar10(2, 1);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("parallel_chains") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_parallel_chains(20, 4);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_parallel_chains(20, 4);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_parallel_chains(20, 4);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp =
            cost_aware_barrier_sync_sp_ization(gv, cost_map);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
      }
    }
  }

  TEST_CASE(
      "dependency_invariant_sp_ization_with_coalescing - cost increases") {
    SUBCASE("Linear Graph") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_linear(10);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_linear(10);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_linear(10);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("fully_connected") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_fully_connected({1, 4, 6, 1});
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_fully_connected({1, 4, 6, 1});
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_fully_connected({1, 4, 6, 1});
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("sample_dag_3") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_sample_dag_3();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_sample_dag_3();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_sample_dag_3();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("sample_dag_1") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_sample_dag_1();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_sample_dag_1();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_sample_dag_1();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("taso_nasnet_cell") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_taso_nasnet_cell();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_taso_nasnet_cell();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_taso_nasnet_cell();
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("cifar10") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_cifar10(2, 1);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_cifar10(2, 1);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_cifar10(2, 1);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
    }
    SUBCASE("parallel_chains") {
      SUBCASE("Constant(1)") {
        DiGraph g = TestingGraphs::make_parallel_chains(4, 2);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Constant(1));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Uniform(0,1)") {
        DiGraph g = TestingGraphs::make_parallel_chains(4, 2);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Uniform(0, 1));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
      SUBCASE("Binary(1,1000)") {
        DiGraph g = TestingGraphs::make_parallel_chains(4, 2);
        DiGraphView gv = flipped(g);
        auto cost_map = Distributions::make_cost_map(
            get_nodes(g), Distributions::Binary(1, 1000));
        SerialParallelDecomposition sp =
            dependency_invariant_sp_ization_with_coalescing(gv);
        CHECK(isclose(relative_cost_increase(gv, sp, cost_map), 1));
        CHECK(
            isclose(relative_critical_path_cost_increase(gv, sp, cost_map), 1));
      }
    }
  }
  TEST_CASE("BenchMark SP-ization Techniques") {
    SUBCASE("linear") {
      DiGraph g = TestingGraphs::make_linear(10);
      DiGraphView gv = flipped(g);

      BenchMark::bench_mark(
          "linear, Constant(1)", gv, Distributions::Constant(1), 100);
      BenchMark::bench_mark(
          "linear, Uniform(0,1)", gv, Distributions::Uniform(0, 1), 100);
      BenchMark::bench_mark(
          "linear, Binary(1, 1000)", gv, Distributions::Binary(1, 1000), 100);
    }

    SUBCASE("sample_dag_3") {
      DiGraph g = TestingGraphs::make_sample_dag_3();
      DiGraphView gv = flipped(g);

      BenchMark::bench_mark(
          "sample_dag_3, Constant(1)", gv, Distributions::Constant(1), 100);
      BenchMark::bench_mark(
          "sample_dag_3, Uniform(0,1)", gv, Distributions::Uniform(0, 1), 100);
      BenchMark::bench_mark("sample_dag_3, Binary(1, 1000)",
                            gv,
                            Distributions::Binary(1, 1000),
                            100);
    }

    SUBCASE("nasnet_cell") {
      DiGraph g = TestingGraphs::make_taso_nasnet_cell();
      DiGraphView gv = flipped(g);

      BenchMark::bench_mark(
          "nasnet_cell, Constant(1)", gv, Distributions::Constant(1), 100);
      BenchMark::bench_mark(
          "nasnet_cell, Uniform(0,1)", gv, Distributions::Uniform(0, 1), 100);
      BenchMark::bench_mark("nasnet_cell, Binary(1, 1000)",
                            gv,
                            Distributions::Binary(1, 1000),
                            100);
    }

    SUBCASE("rhombus") {
      DiGraph g = TestingGraphs::make_rhombus();
      DiGraphView gv = flipped(g);

      BenchMark::bench_mark(
          "rhombus, Constant(1)", gv, Distributions::Constant(1), 100);
      BenchMark::bench_mark(
          "rhombus, Uniform(0,1)", gv, Distributions::Uniform(0, 1), 100);
      BenchMark::bench_mark(
          "rhombus, Binary(1, 1000)", gv, Distributions::Binary(1, 1000), 100);
    }

    SUBCASE("diamond") {
      DiGraph g = TestingGraphs::make_diamond();
      DiGraphView gv = flipped(g);

      BenchMark::bench_mark(
          "diamond, Constant(1)", gv, Distributions::Constant(1), 100);
      BenchMark::bench_mark(
          "diamond, Uniform(0,1)", gv, Distributions::Uniform(0, 1), 100);
      BenchMark::bench_mark(
          "diamond, Binary(1, 1000)", gv, Distributions::Binary(1, 1000), 100);
    }

    SUBCASE("fully_connected") {
      DiGraph g = TestingGraphs::make_fully_connected({1, 4, 6, 1});
      DiGraphView gv = flipped(g);

      BenchMark::bench_mark(
          "fully_connected, Constant(1)", gv, Distributions::Constant(1), 100);
      BenchMark::bench_mark("fully_connected, Uniform(0,1)",
                            gv,
                            Distributions::Uniform(0, 1),
                            100);
      BenchMark::bench_mark("fully_connected, Binary(1, 1000)",
                            gv,
                            Distributions::Binary(1, 1000),
                            100);
    }

    SUBCASE("parallel_chains") {
      DiGraph g = TestingGraphs::make_parallel_chains(20, 4);
      DiGraphView gv = flipped(g);

      BenchMark::bench_mark(
          "parallel_chains, Constant(1)", gv, Distributions::Constant(1), 100);
      BenchMark::bench_mark("parallel_chains, Uniform(0,1)",
                            gv,
                            Distributions::Uniform(0, 1),
                            100);
      BenchMark::bench_mark("parallel_chains, Binary(1, 1000)",
                            gv,
                            Distributions::Binary(1, 1000),
                            100);
    }

    SUBCASE("cifar10") {
      DiGraph g = TestingGraphs::make_cifar10(1, 1);
      DiGraphView gv = flipped(g);

      BenchMark::bench_mark(
          "cifar10, Constant(1)", gv, Distributions::Constant(1), 5);
      BenchMark::bench_mark(
          "cifar10, Uniform(0,1)", gv, Distributions::Uniform(0, 1), 5);
      BenchMark::bench_mark(
          "cifar10, Binary(1, 1000)", gv, Distributions::Binary(1, 1000), 5);
    }

    SUBCASE("sample_dag_1") {
      DiGraph g = TestingGraphs::make_sample_dag_1();
      DiGraphView gv = flipped(g);

      BenchMark::bench_mark(
          "sample_dag_1, Constant(1)", gv, Distributions::Constant(1), 100);
      BenchMark::bench_mark(
          "sample_dag_1, Uniform(0,1)", gv, Distributions::Uniform(0, 1), 100);
      BenchMark::bench_mark("sample_dag_1, Binary(1, 1000)",
                            gv,
                            Distributions::Binary(1, 1000),
                            100);
    }
  }
}
