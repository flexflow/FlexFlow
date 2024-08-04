#include "utils/graph/serial_parallel/serial_parallel_isempty.h"
#include "utils/containers/all_of.h"
#include "utils/containers/transform.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include "utils/variant.h"

namespace FlexFlow {

bool isempty(SerialParallelDecomposition const &sp) {
  if (sp.has<Node>()) {
    return false;
  } else if (sp.has<SerialSplit>()) {
    return all_of(transform(sp.get<SerialSplit>().children,
                            [](auto const &child) {
                              return widen<SerialParallelDecomposition>(child);
                            }),
                  isempty);
  } else {
    assert(sp.has<ParallelSplit>());
    return all_of(transform(sp.get<ParallelSplit>().children,
                            [](auto const &child) {
                              return widen<SerialParallelDecomposition>(child);
                            }),
                  isempty);
  }
}

} // namespace FlexFlow
