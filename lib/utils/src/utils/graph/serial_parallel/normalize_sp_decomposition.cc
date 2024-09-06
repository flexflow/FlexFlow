#include "utils/graph/serial_parallel/normalize_sp_decomposition.h"
#include "utils/containers/filter.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"
#include "utils/variant.h"

namespace FlexFlow {

template <typename T>
static auto filter_empty(T const &container) {
  return filter(container, [](auto const &child) {
    return !is_empty(widen<SerialParallelDecomposition>(child));
  });
}

SerialParallelDecomposition normalize_sp_decomposition(Node const &node) {
  return SerialParallelDecomposition(node);
}

SerialParallelDecomposition
    normalize_sp_decomposition(SerialSplit const &serial) {
  std::vector<SerialParallelDecomposition> normalized_children =
      transform(filter_empty(serial.children), [](auto const &child) {
        return normalize_sp_decomposition(
            widen<SerialParallelDecomposition>(child));
      });

  if (normalized_children.size() == 1) {
    return get_only(normalized_children);
  }
  return serial_composition(normalized_children);
}

SerialParallelDecomposition
    normalize_sp_decomposition(ParallelSplit const &parallel) {
  std::unordered_set<SerialParallelDecomposition> normalized_children =
      transform(filter_empty(parallel.children), [](auto const &child) {
        return normalize_sp_decomposition(
            widen<SerialParallelDecomposition>(child));
      });

  if (normalized_children.size() == 1) {
    return get_only(normalized_children);
  }
  return parallel_composition(normalized_children);
}

SerialParallelDecomposition
    normalize_sp_decomposition(SerialParallelDecomposition const &sp) {
  return sp.visit<SerialParallelDecomposition>(
      [](auto const &x) { return normalize_sp_decomposition(x); });
}

} // namespace FlexFlow
