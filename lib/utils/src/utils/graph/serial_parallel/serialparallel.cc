#include "utils/graph/serial_parallel/serialparallel.h"
#include "utils/containers/transform.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"
#include "utils/graph/serial_parallel/serialparallel_internal.h"
#include "utils/graph/serial_parallel/sink_settings.dtg.h"
#include "utils/graph/serial_parallel/source_settings.dtg.h"

namespace FlexFlow {

std::optional<SerialParallelDecomposition>
    get_serial_parallel_decomposition(DiGraphView const &g) {
  return transform(
      sp_decomposition(g),
      [](std::variant<IntermediateSpDecompositionTree, Node> const &ast) {
        return to_final_ast(ast);
      });
}

} // namespace FlexFlow
