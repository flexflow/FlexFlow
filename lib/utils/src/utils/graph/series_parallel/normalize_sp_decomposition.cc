#include "utils/graph/series_parallel/normalize_sp_decomposition.h"
#include "utils/containers/filter.h"
#include "utils/containers/get_only.h"
#include "utils/containers/multiset_of.h"
#include "utils/containers/transform.h"
#include "utils/graph/series_parallel/non_normal_sp_decomposition.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.h"
#include "utils/variant.h"

namespace FlexFlow {

template <typename T>
static auto filter_empty(T const &container) {
  return filter(container, [](auto const &child) {
    return !is_empty_non_normal(widen<NonNormalSPDecomposition>(child));
  });
}

static SeriesParallelDecomposition
    normalize_sp_decomposition(Node const &node) {
  return SeriesParallelDecomposition(node);
}

static SeriesParallelDecomposition
    normalize_sp_decomposition(NonNormalSeriesSplit const &serial) {
  std::vector<SeriesParallelDecomposition> normalized_children =
      transform(filter_empty(serial.children),
                [](std::variant<NonNormalParallelSplit, Node> const &child) {
                  return normalize_sp_decomposition(
                      widen<NonNormalSPDecomposition>(child));
                });

  ASSERT(normalized_children.size() > 0, "Cannot normalize empty SeriesSplit");

  if (normalized_children.size() == 1) {
    return get_only(normalized_children);
  }

  return series_composition(normalized_children);
}

static SeriesParallelDecomposition
    normalize_sp_decomposition(NonNormalParallelSplit const &parallel) {
  std::multiset<SeriesParallelDecomposition> normalized_children =
      transform(filter_empty(parallel.get_children()),
                [](std::variant<NonNormalSeriesSplit, Node> const &child) {
                  return normalize_sp_decomposition(
                      widen<NonNormalSPDecomposition>(child));
                });

  ASSERT(normalized_children.size() > 0,
         "Cannot normalize empty ParallelSplit");

  if (normalized_children.size() == 1) {
    return get_only(normalized_children);
  }
  return parallel_composition(multiset_of(normalized_children));
}

SeriesParallelDecomposition
    normalize_sp_decomposition(NonNormalSPDecomposition const &sp) {
  return sp.visit<SeriesParallelDecomposition>(
      [](auto const &t) { return normalize_sp_decomposition(t); });
}

} // namespace FlexFlow
