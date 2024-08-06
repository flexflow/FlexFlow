#include "utils/graph/serial_parallel/serial_parallel_normalize.h"
#include "utils/containers/filter.h"
#include "utils/containers/get_only.h"
#include "utils/containers/transform.h"
#include "utils/graph/serial_parallel/parallel_composition.h"
#include "utils/graph/serial_parallel/serial_composition.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include "utils/graph/serial_parallel/serial_parallel_isempty.h"
#include "utils/variant.h"

namespace FlexFlow {

static std::vector<std::variant<ParallelSplit, Node>>
    filter_empty(SerialSplit const &serial) {
  return filter(serial.children, [](auto const &child) {
    return !isempty(widen<SerialParallelDecomposition>(child));
  });
}

static std::unordered_set<std::variant<SerialSplit, Node>>
    filter_empty(ParallelSplit const &parallel) {
  return filter(parallel.children, [](auto const &child) {
    return !isempty(widen<SerialParallelDecomposition>(child));
  });
}

SerialParallelDecomposition normalize(SerialParallelDecomposition const &sp) {

  auto normalize_children = [](auto const &container) {
    return transform(filter_empty(container), [](auto const child) {
      return normalize(widen<SerialParallelDecomposition>(child));
    });
  };

  auto simplify_composition =
      [](SerialParallelDecomposition const &composition) {
        if (composition.has<SerialSplit>()) {
          SerialSplit serial = composition.get<SerialSplit>();
          if (serial.children.size() == 1) {
            return widen<SerialParallelDecomposition>(
                get_only(serial.children));
          }
        } else if (composition.has<ParallelSplit>()) {
          ParallelSplit parallel = composition.get<ParallelSplit>();
          if (parallel.children.size() == 1) {
            return widen<SerialParallelDecomposition>(
                get_only(parallel.children));
          }
        }
        return composition;
      };

  if (sp.has<Node>()) {
    return sp;
  }
  if (sp.has<SerialSplit>()) {
    std::vector<SerialParallelDecomposition> normalized_children =
        normalize_children(sp.get<SerialSplit>());
    return simplify_composition(serial_composition(normalized_children));
  } else {
    assert(sp.has<ParallelSplit>());
    std::unordered_set<SerialParallelDecomposition> normalized_children =
        normalize_children(sp.get<ParallelSplit>());
    return simplify_composition(parallel_composition(normalized_children));
  }
}

} // namespace FlexFlow
