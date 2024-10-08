#include "utils/graph/series_parallel/binary_sp_decomposition_tree/right_associative_binary_sp_tree_from_nary.h"
#include "utils/containers/foldr1.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include "utils/overload.h"

namespace FlexFlow {

BinarySPDecompositionTree right_associative_binary_sp_tree_from_nary(
    SeriesParallelDecomposition const &nary) {
  std::function<BinarySPDecompositionTree(
      std::variant<ParallelSplit, Node> const &)>
      from_series_child;
  std::function<BinarySPDecompositionTree(
      std::variant<SeriesSplit, Node> const &)>
      from_parallel_child;

  auto from_node = [](Node const &n) { return BinarySPDecompositionTree{n}; };

  auto from_series = [&](SeriesSplit const &s) {
    std::vector<BinarySPDecompositionTree> children =
        transform(s.children, from_series_child);
    return foldr1(
        children,
        [](BinarySPDecompositionTree const &accum,
           BinarySPDecompositionTree const &x) -> BinarySPDecompositionTree {
          return BinarySPDecompositionTree{
              BinarySeriesSplit{x, accum},
          };
        });
  };

  auto from_parallel = [&](ParallelSplit const &s) {
    std::vector<BinarySPDecompositionTree> children =
        transform(vector_of(s.get_children()), from_parallel_child);
    return foldr1(
        children,
        [](BinarySPDecompositionTree const &accum,
           BinarySPDecompositionTree const &x) -> BinarySPDecompositionTree {
          return BinarySPDecompositionTree{
              BinaryParallelSplit{x, accum},
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
