#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_PARALLEL_COMPOSITION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_PARALLEL_COMPOSITION_H

#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"

namespace FlexFlow {

SerialParallelDecomposition parallel_composition(
    std::vector<SerialParallelDecomposition> const &sp_compositions);

} // namespace FlexFlow

#endif
