#include "utils/graph/serial_parallel/parallel_composition.h"
#include "utils/containers/get_only.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"

namespace FlexFlow {

SerialParallelDecomposition parallel_composition(
    std::unordered_set<SerialParallelDecomposition> const &sp_compositions) {
  if (sp_compositions.size() == 1) {
    return get_only(sp_compositions);
  }
  ParallelSplit composition({});
  for (SerialParallelDecomposition const &sp_comp : sp_compositions) {
    if (sp_comp.has<ParallelSplit>()) {
      for (std::variant<SerialSplit, Node> const &children :
           sp_comp.get<ParallelSplit>()
               .children) { // unwrapping the parallel node, since a
                            // ParallelSplit cannot contain other Parallels
        composition.children.insert(children);
      }
    } else if (sp_comp.has<SerialSplit>()) {
      composition.children.insert(sp_comp.get<SerialSplit>());
    } else {
      assert(sp_comp.has<Node>());
      composition.children.insert(sp_comp.get<Node>());
    }
  }
  return SerialParallelDecomposition(composition);
}

} // namespace FlexFlow
