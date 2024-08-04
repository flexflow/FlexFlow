#include "utils/containers/get_only.h"
#include "utils/graph/serial_parallel/parallel_composition.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"

namespace FlexFlow {

SerialParallelDecomposition serial_composition(
    std::vector<SerialParallelDecomposition> const &sp_compositions) {
  if (sp_compositions.size() == 1) {
    return get_only(sp_compositions);
  }
  SerialSplit composition{};
  for (SerialParallelDecomposition const &sp_comp : sp_compositions) {
    if (sp_comp.has<SerialSplit>()) {
      for (std::variant<ParallelSplit, Node> const &subnode :
           sp_comp.get<SerialSplit>()
               .children) { // unwrapping the serial node, since a SerialSplit
                            // cannot contain other Serials
        composition.children.push_back(subnode);
      }
    } else if (sp_comp.has<ParallelSplit>()) {
      composition.children.push_back(sp_comp.get<ParallelSplit>());
    } else {
      assert(sp_comp.has<Node>());
      composition.children.push_back(sp_comp.get<Node>());
    }
  }
  return SerialParallelDecomposition(composition);
}

} // namespace FlexFlow
