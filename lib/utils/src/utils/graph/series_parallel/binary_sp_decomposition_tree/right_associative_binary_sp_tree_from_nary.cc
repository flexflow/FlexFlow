#include "utils/graph/series_parallel/binary_sp_decomposition_tree/right_associative_binary_sp_tree_from_nary.h"
#include "utils/containers/foldr1.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/overload.h"

namespace FlexFlow {

BinarySPDecompositionTree right_associative_binary_sp_tree_from_nary(
    SeriesParallelDecomposition const &nary) {
  std::function<GenericBinarySPDecompositionTree<Node>(
      std::variant<ParallelSplit, Node> const &)>
      from_series_child;
  std::function<GenericBinarySPDecompositionTree<Node>(
      std::variant<SeriesSplit, Node> const &)>
      from_parallel_child;

  auto from_node = [](Node const &n) {
    return GenericBinarySPDecompositionTree<Node>{n};
  };

  auto from_series = [&](SeriesSplit const &s) {
    std::vector<GenericBinarySPDecompositionTree<Node>> children =
        transform(s.children, from_series_child);
    return foldr1(children,
                  [](GenericBinarySPDecompositionTree<Node> const &accum,
                     GenericBinarySPDecompositionTree<Node> const &x) {
                    return GenericBinarySPDecompositionTree<Node>{
                        GenericBinarySeriesSplit<Node>{x, accum}};
                  });
  };

  auto from_parallel = [&](ParallelSplit const &s) {
    std::vector<GenericBinarySPDecompositionTree<Node>> children =
        transform(vector_of(s.children), from_parallel_child);
    return foldr1(children,
                  [](GenericBinarySPDecompositionTree<Node> const &accum,
                     GenericBinarySPDecompositionTree<Node> const &x) {
                    return GenericBinarySPDecompositionTree<Node>{
                        GenericBinaryParallelSplit<Node>{x, accum}};
                  });
  };

  from_parallel_child = [&](std::variant<SeriesSplit, Node> const &v)
      -> GenericBinarySPDecompositionTree<Node> {
    return std::visit(overload{
                          [&](Node const &n) { return from_node(n); },
                          [&](SeriesSplit const &s) { return from_series(s); },
                      },
                      v);
  };

  from_series_child = [&](std::variant<ParallelSplit, Node> const &v)
      -> GenericBinarySPDecompositionTree<Node> {
    return std::visit(
        overload{
            [&](Node const &n) { return from_node(n); },
            [&](ParallelSplit const &p) { return from_parallel(p); },
        },
        v);
  };

  return BinarySPDecompositionTree{
      nary.visit<GenericBinarySPDecompositionTree<Node>>(overload{
          [&](Node const &n) { return from_node(n); },
          [&](SeriesSplit const &s) { return from_series(s); },
          [&](ParallelSplit const &p) { return from_parallel(p); },
      }),
  };
}

} // namespace FlexFlow
