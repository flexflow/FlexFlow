#include "utils/graph/series_parallel/binary_sp_decomposition_tree/left_associative_binary_sp_tree_from_nary.h"
#include "utils/containers/foldl1.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include "utils/overload.h"

namespace FlexFlow {

BinarySPDecompositionTree left_associative_binary_sp_tree_from_nary(
    SeriesParallelDecomposition const &nary) {
  std::function<BinarySPDecompositionTree(
      std::variant<ParallelSplit, Node> const &)>
      from_series_child;
  std::function<BinarySPDecompositionTree(
      std::variant<SeriesSplit, Node> const &)>
      from_parallel_child;

  auto from_node = [](Node const &n) -> BinarySPDecompositionTree {
    return BinarySPDecompositionTree{n};
  };

  auto from_series = [&](SeriesSplit const &s) -> BinarySPDecompositionTree {
    std::vector<BinarySPDecompositionTree> children =
        transform(s.children, from_series_child);
    return foldl1(children,
                  [](BinarySPDecompositionTree const &accum,
                     BinarySPDecompositionTree const &x) -> BinarySPDecompositionTree {
                    return BinarySPDecompositionTree{
                      BinarySeriesSplit{accum, x},
                    };
                  });
  };

  auto from_parallel =
      [&](ParallelSplit const &s) -> BinarySPDecompositionTree {
    std::vector<BinarySPDecompositionTree> children =
        transform(vector_of(s.get_children()), from_parallel_child);
    return foldl1(children,
                  [](BinarySPDecompositionTree const &accum,
                     BinarySPDecompositionTree const &x) -> BinarySPDecompositionTree {
                    return BinarySPDecompositionTree{
                      BinaryParallelSplit{accum, x},
                    };
                  });
  };

  from_parallel_child = [&](std::variant<SeriesSplit, Node> const &v)
      -> BinarySPDecompositionTree {
    return std::visit(overload{
                          [&](Node const &n) { return from_node(n); },
                          [&](SeriesSplit const &s) { return from_series(s); },
                      },
                      v);
  };

  from_series_child = [&](std::variant<ParallelSplit, Node> const &v)
      -> BinarySPDecompositionTree {
    return std::visit(
        overload{
            [&](Node const &n) { return from_node(n); },
            [&](ParallelSplit const &p) { return from_parallel(p); },
        },
        v);
  };

  return nary.visit<BinarySPDecompositionTree>(overload{
      [&](Node const &n) { return from_node(n); },
      [&](SeriesSplit const &s) { return from_series(s); },
      [&](ParallelSplit const &p) { return from_parallel(p); },
  });
}

} // namespace FlexFlow
