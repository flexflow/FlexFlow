#include "utils/graph/serial_parallel/serialparallel.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.h"
#include "utils/graph/serial_parallel/serialparallel_internal.h"
#include "utils/graph/serial_parallel/sink_settings.dtg.h"
#include "utils/graph/serial_parallel/source_settings.dtg.h"
#include "utils/containers/transform.h"
#include "utils/graph/algorithms.h"

namespace FlexFlow {

SerialParallelDecomposition
    get_serial_parallel_decomposition(DiGraphView const &g) {
  std::variant<IntermediateSpDecompositionTree, Node> ast = sp_decomposition(g);
  return to_final_ast(ast);
}

} // namespace FlexFlow
