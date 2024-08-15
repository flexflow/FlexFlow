// For context, see https://arxiv.org/abs/1902.09635 &&
// https://github.com/google-research/nasbench/blob/master/nasbench/api.py

#include "utils/containers.h"
#include "utils/containers/all_of.h"
#include "utils/containers/repeat.h"
#include "utils/containers/transform.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/digraph/algorithms/is_acyclic.h"
#include "utils/graph/digraph/algorithms/materialize_digraph_view.h"
#include "utils/graph/digraph/algorithms/transitive_reduction.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/serial_parallel/digraph_generation.h"
#include <optional>
#include <vector>

constexpr size_t MIN_NODES = 6;
constexpr size_t MAX_NODES = 8;
constexpr size_t MIN_EDGES = 8;
constexpr size_t MAX_EDGES = 11;
constexpr size_t NUM_CELLS = 9;

using AdjacencyMatrix = std::vector<std::vector<bool>>;
namespace FlexFlow {
struct NasNetBenchConfig {
  AdjacencyMatrix adjacency_matrix;
};

bool is_valid_config(NasNetBenchConfig const &config) {
  AdjacencyMatrix const &matrix = config.adjacency_matrix;
  const size_t size = matrix.size();

  auto is_valid_size = [](size_t s) {
    return s >= MIN_NODES && s <= MAX_NODES;
  };

  auto is_square_matrix = [&](auto const &m) {
    return all_of(m, [&](const auto &row) { return row.size() == size; });
  };

  auto is_upper_triangular = [&](auto const &m) {
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j <= i; ++j) {
        if (matrix[i][j]) {
          return false;
        }
      }
    }
    return true;
  };

  return is_valid_size(size) && is_square_matrix(matrix) &&
         is_upper_triangular(matrix);
}

bool is_valid_cell(DiGraphView const &g) {
  return (is_acyclic(g)) && (get_sources(g).size() == 1) &&
         (get_sinks(g).size() == 1) && (num_edges(g) <= MAX_EDGES) &&
         (num_edges(g) >= MIN_EDGES) && (num_edges(g) <= MAX_NODES) &&
         (num_edges(g) >= MIN_NODES) &&
         (num_edges(g) > num_nodes(g)); // filter linear cell and diamond cell
}

NasNetBenchConfig generate_random_config() {
  static std::uniform_int_distribution<> size_dist(MIN_NODES, MAX_NODES);
  Binary bin = Binary(0, 1);

  size_t num_nodes = Uniform(MIN_NODES, MAX_NODES)();
  std::vector<std::vector<bool>> matrix(num_nodes,
                                        std::vector<bool>(num_nodes, false));

  for (size_t i = 0; i < num_nodes; ++i) {
    for (size_t j = i + 1; j < num_nodes; ++j) {
      matrix[i][j] = bin();
    }
  }

  return {matrix};
}

std::optional<DiGraph>
    maybe_generate_nasnet_bench_cell(NasNetBenchConfig const &config) {
  if (!is_valid_config(config)) {
    return std::nullopt;
  }

  DiGraph g = DiGraph::create<AdjacencyDiGraph>();
  std::vector<Node> nodes = add_nodes(g, config.adjacency_matrix.size());

  for (size_t i = 0; i < nodes.size(); ++i) {
    for (size_t j = i + 1; j < nodes.size(); ++j) {
      if (config.adjacency_matrix[i][j]) {
        g.add_edge(DirectedEdge{nodes[i], nodes[j]});
      }
    }
  }

  g = materialize_digraph_view<AdjacencyDiGraph>(transitive_reduction(g));

  if (!is_valid_cell(g)) {
    return std::nullopt;
  }

  return g;
}

DiGraph generate_nasnet_bench_cell() {
  while (true) {
    NasNetBenchConfig config = generate_random_config();
    std::optional<DiGraph> maybe_cell =
        maybe_generate_nasnet_bench_cell(config);
    if (maybe_cell) {
      return maybe_cell.value();
    }
  }
}

DiGraph generate_nasnet_bench_network() {
  DiGraph g = serial_composition(
      transform(repeat(NUM_CELLS, generate_nasnet_bench_cell),
                [](auto const cell) -> DiGraphView { return cell; }));
  return g;
}
} // namespace FlexFlow
