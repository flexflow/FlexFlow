#include "distributions.h"
#include "sample_graphs.h"
#include "test/utils/doctest.h"
#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include "utils/graph/serial_parallel/serial_parallel_metrics.h"
#include "utils/graph/serial_parallel/sp_ization/critical_path_preserving_sp_ization.h"
#include "utils/graph/serial_parallel/sp_ization/work_preserving_sp_ization.h"
#include <iostream>
#include <string>
#include <tuple>

using namespace FlexFlow;

using Result = std::pair<float, float>;
using CombinedResult = std::tuple<Result, Result, Result>;
template <typename D, typename N = NoNoise>
CombinedResult perform_benchmark(DiGraphView const &g,
                                 D const &Dist,
                                 N const &Noise = NoNoise(),
                                 size_t const &repeat = 100) {
  Result critical_path_preserving = {0, 0};
  Result barrier_sync = {0, 0};
  Result cost_aware = {0, 0};
  for (int i = 0; i < repeat; i++) {
    auto cost_map = make_cost_map(get_nodes(g), Dist, Noise);
    SerialParallelDecomposition sp1 =
        critical_path_preserving_sp_ization_with_coalescing(g);
    SerialParallelDecomposition sp2 = barrier_sync_sp_ization(g);
    SerialParallelDecomposition sp3 =
        cost_aware_barrier_sync_sp_ization(g, cost_map);

    critical_path_preserving.first += relative_work_increase(g, sp1, cost_map);
    critical_path_preserving.second +=
        relative_critical_path_cost_increase(g, sp1, cost_map);
    barrier_sync.first += relative_work_increase(g, sp2, cost_map);
    barrier_sync.second +=
        relative_critical_path_cost_increase(g, sp2, cost_map);
    cost_aware.first += relative_work_increase(g, sp3, cost_map);
    cost_aware.second += relative_critical_path_cost_increase(g, sp3, cost_map);
  }
  std::vector<Result> results = {
      critical_path_preserving, barrier_sync, cost_aware};
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
  std::cout << "Barrier Sync         | " << b_s.first << " | " << b_s.second
            << std::endl;
  std::cout << "Cost Aware           | " << c_a.first << " | " << c_a.second
            << std::endl;
  std::cout << "Dependency Invariant | " << d_i.first << " | " << d_i.second
            << std::endl;
  std::cout << std::endl;
}

template <typename D, typename N = NoNoise>
void bench_mark(std::string title,
                DiGraphView const &g,
                D const &Dist,
                N const &Noise = NoNoise(),
                size_t const &repeat = 100) {
  output_benchmark(perform_benchmark(g, Dist, Noise, repeat), title);
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("BenchMark SP-ization Techniques") {

    SUBCASE("sample_dag_3") {
      DiGraph g = make_sample_dag_3();

      bench_mark("sample_dag_3, Constant(1)", g, Constant(1));
      bench_mark("sample_dag_3, Constant(1), UniformNoise(0.8, 1.25)",
                 g,
                 Constant(1),
                 UniformNoise(0.8, 1.25));
      bench_mark("sample_dag_3, Constant(1), GaussianNoise(1, 0.1)",
                 g,
                 Constant(1),
                 GaussianNoise(1, 0.1));

      bench_mark("sample_dag_3, Uniform(0,1)", g, Uniform(0, 1));
      bench_mark("sample_dag_3, Uniform(0,1), UniformNoise(0.8, 1.25)",
                 g,
                 Uniform(0, 1),
                 UniformNoise(0.8, 1.25));
      bench_mark("sample_dag_3, Uniform(0,1), GaussianNoise(1, 0.1)",
                 g,
                 Uniform(0, 1),
                 GaussianNoise(1, 0.1));

      bench_mark("sample_dag_3, Binary(1, 1000)", g, Binary(1, 1000));
      bench_mark("sample_dag_3, Binary(1, 1000), UniformNoise(0.8, 1.25)",
                 g,
                 Binary(1, 1000),
                 UniformNoise(0.8, 1.25));
      bench_mark("sample_dag_3, Binary(1, 1000), GaussianNoise(1, 0.1)",
                 g,
                 Binary(1, 1000),
                 GaussianNoise(1, 0.1));
    }

    SUBCASE("nasnet_cell") {
      DiGraph g = make_taso_nasnet_cell();

      bench_mark("nasnet_cell, Constant(1)", g, Constant(1));
      bench_mark("nasnet_cell, Constant(1), UniformNoise(0.8, 1.25)",
                 g,
                 Constant(1),
                 UniformNoise(0.8, 1.25));
      bench_mark("nasnet_cell, Constant(1), GaussianNoise(1, 0.1)",
                 g,
                 Constant(1),
                 GaussianNoise(1, 0.1));

      bench_mark("nasnet_cell, Uniform(0,1)", g, Uniform(0, 1));
      bench_mark("nasnet_cell, Uniform(0,1), UniformNoise(0.8, 1.25)",
                 g,
                 Uniform(0, 1),
                 UniformNoise(0.8, 1.25));
      bench_mark("nasnet_cell, Uniform(0,1), GaussianNoise(1, 0.1)",
                 g,
                 Uniform(0, 1),
                 GaussianNoise(1, 0.1));

      bench_mark("nasnet_cell, Binary(1, 1000)", g, Binary(1, 1000));
      bench_mark("nasnet_cell, Binary(1, 1000), UniformNoise(0.8, 1.25)",
                 g,
                 Binary(1, 1000),
                 UniformNoise(0.8, 1.25));
      bench_mark("nasnet_cell, Binary(1, 1000), GaussianNoise(1, 0.1)",
                 g,
                 Binary(1, 1000),
                 GaussianNoise(1, 0.1));
    }

    SUBCASE("fully_connected") {
      DiGraph g = make_fully_connected({1, 4, 6, 4, 1});

      bench_mark("fully_connected, Constant(1)", g, Constant(1));
      bench_mark("fully_connected, Constant(1), UniformNoise(0.8, 1.25)",
                 g,
                 Constant(1),
                 UniformNoise(0.8, 1.25));
      bench_mark("fully_connected, Constant(1), GaussianNoise(1, 0.1)",
                 g,
                 Constant(1),
                 GaussianNoise(1, 0.1));

      bench_mark("fully_connected, Uniform(0,1)", g, Uniform(0, 1));
      bench_mark("fully_connected, Uniform(0,1), UniformNoise(0.8, 1.25)",
                 g,
                 Uniform(0, 1),
                 UniformNoise(0.8, 1.25));
      bench_mark("fully_connected, Uniform(0,1), GaussianNoise(1, 0.1)",
                 g,
                 Uniform(0, 1),
                 GaussianNoise(1, 0.1));

      bench_mark("fully_connected, Binary(1, 1000)", g, Binary(1, 1000));
      bench_mark("fully_connected, Binary(1, 1000), UniformNoise(0.8, 1.25)",
                 g,
                 Binary(1, 1000),
                 UniformNoise(0.8, 1.25));
      bench_mark("fully_connected, Binary(1, 1000), GaussianNoise(1, 0.1)",
                 g,
                 Binary(1, 1000),
                 GaussianNoise(1, 0.1));
    }

    SUBCASE("parallel_chains") {
      DiGraph g = make_parallel_chains(8, 3);

      bench_mark("parallel_chains, Constant(1)", g, Constant(1));
      bench_mark("parallel_chains, Constant(1), UniformNoise(0.8, 1.25)",
                 g,
                 Constant(1),
                 UniformNoise(0.8, 1.25));
      bench_mark("parallel_chains, Constant(1), GaussianNoise(1, 0.1)",
                 g,
                 Constant(1),
                 GaussianNoise(1, 0.1));

      bench_mark("parallel_chains, Uniform(0,1)", g, Uniform(0, 1));
      bench_mark("parallel_chains, Uniform(0,1), UniformNoise(0.8, 1.25)",
                 g,
                 Uniform(0, 1),
                 UniformNoise(0.8, 1.25));
      bench_mark("parallel_chains, Uniform(0,1), GaussianNoise(1, 0.1)",
                 g,
                 Uniform(0, 1),
                 GaussianNoise(1, 0.1));

      bench_mark("parallel_chains, Binary(1, 1000)", g, Binary(1, 1000));
      bench_mark("parallel_chains, Binary(1, 1000), UniformNoise(0.8, 1.25)",
                 g,
                 Binary(1, 1000),
                 UniformNoise(0.8, 1.25));
      bench_mark("parallel_chains, Binary(1, 1000), GaussianNoise(1, 0.1)",
                 g,
                 Binary(1, 1000),
                 GaussianNoise(1, 0.1));
    }

    SUBCASE("cifar10") {
      DiGraph g = make_cifar10(2, 1);

      bench_mark("cifar10, Constant(1)", g, Constant(1), NoNoise(), 10);
      bench_mark("cifar10, Constant(1), UniformNoise(0.8, 1.25)",
                 g,
                 Constant(1),
                 UniformNoise(0.8, 1.25),
                 10);
      bench_mark("cifar10, Constant(1), GaussianNoise(1, 0.1)",
                 g,
                 Constant(1),
                 GaussianNoise(1, 0.1),
                 10);

      bench_mark("cifar10, Uniform(0,1)", g, Uniform(0, 1), NoNoise(), 10);
      bench_mark("cifar10, Uniform(0,1), UniformNoise(0.8, 1.25)",
                 g,
                 Uniform(0, 1),
                 UniformNoise(0.8, 1.25),
                 10);
      bench_mark("cifar10, Uniform(0,1), GaussianNoise(1, 0.1)",
                 g,
                 Uniform(0, 1),
                 GaussianNoise(1, 0.1),
                 10);

      bench_mark("cifar10, Binary(1, 1000)", g, Binary(1, 1000), NoNoise(), 10);
      bench_mark("cifar10, Binary(1, 1000), UniformNoise(0.8, 1.25)",
                 g,
                 Binary(1, 1000),
                 UniformNoise(0.8, 1.25),
                 10);
      bench_mark("cifar10, Binary(1, 1000), GaussianNoise(1, 0.1)",
                 g,
                 Binary(1, 1000),
                 GaussianNoise(1, 0.1),
                 10);
    }

    SUBCASE("random_2_terminal_random_dag") {
      DiGraph g = make_2_terminal_random_dag(100, .1, 10);

      bench_mark("random_2_terminal_random_dag, Constant(1)",
                 g,
                 Constant(1),
                 NoNoise(),
                 10);
      bench_mark(
          "random_2_terminal_random_dag, Constant(1), UniformNoise(0.8, 1.25)",
          g,
          Constant(1),
          UniformNoise(0.8, 1.25),
          10);
      bench_mark(
          "random_2_terminal_random_dag, Constant(1), GaussianNoise(1, 0.1)",
          g,
          Constant(1),
          GaussianNoise(1, 0.1),
          10);

      bench_mark("random_2_terminal_random_dag, Uniform(0,1)",
                 g,
                 Uniform(0, 1),
                 NoNoise(),
                 10);
      bench_mark(
          "random_2_terminal_random_dag, Uniform(0,1), UniformNoise(0.8, 1.25)",
          g,
          Uniform(0, 1),
          UniformNoise(0.8, 1.25),
          10);
      bench_mark(
          "random_2_terminal_random_dag, Uniform(0,1), GaussianNoise(1, 0.1)",
          g,
          Uniform(0, 1),
          GaussianNoise(1, 0.1),
          10);

      bench_mark("random_2_terminal_random_dag, Binary(1, 1000)",
                 g,
                 Binary(1, 1000),
                 NoNoise(),
                 10);
      bench_mark("random_2_terminal_random_dag, Binary(1, 1000), "
                 "UniformNoise(0.8, 1.25)",
                 g,
                 Binary(1, 1000),
                 UniformNoise(0.8, 1.25),
                 10);
      bench_mark("random_2_terminal_random_dag, Binary(1, 1000), "
                 "GaussianNoise(1, 0.1)",
                 g,
                 Binary(1, 1000),
                 GaussianNoise(1, 0.1),
                 10);
    }
  }
}
