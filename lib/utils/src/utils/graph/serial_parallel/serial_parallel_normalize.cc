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
  // return filter(serial.children, [](auto const &child) {return
  // !isempty(widen<SerialParallelDecomposition>(child));});
  std::vector<std::variant<ParallelSplit, Node>> filtered;
  for (std::variant<ParallelSplit, Node> const &child : serial.children) {
    if (!isempty(widen<SerialParallelDecomposition>(child))) {
      filtered.push_back(child);
    }
  }
  return filtered;
}

static std::vector<std::variant<SerialSplit, Node>>
    filter_empty(ParallelSplit const &parallel) {
  // return filter(parallel.children, [](auto const &child) {return
  // !isempty(widen<SerialParallelDecomposition>(child));});
  std::vector<std::variant<SerialSplit, Node>> filtered;
  for (std::variant<SerialSplit, Node> const &child : parallel.children) {
    if (!isempty(widen<SerialParallelDecomposition>(child))) {
      filtered.push_back(child);
    }
  }
  return filtered;
}

SerialParallelDecomposition normalize(SerialParallelDecomposition const &sp) {

  auto normalize_children = [](auto const &container) {
    std::vector<SerialParallelDecomposition> normalized_children;
    for (const auto &child : filter_empty(container)) {
      if (std::holds_alternative<Node>(child)) {
        normalized_children.push_back(widen<SerialParallelDecomposition>(
            child)); // TODO make the function a single transform
      } else {
        normalized_children.push_back(
            normalize(widen<SerialParallelDecomposition>(child)));
      }
    }
    return normalized_children;
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
    std::vector<SerialParallelDecomposition> normalized_children =
        normalize_children(sp.get<ParallelSplit>());
    return simplify_composition(parallel_composition(normalized_children));
  }
}

} // namespace FlexFlow
