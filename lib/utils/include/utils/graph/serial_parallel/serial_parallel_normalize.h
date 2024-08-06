#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_NORMALIZE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIAL_PARALLEL_NORMALIZE_H

#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"

namespace FlexFlow {

/**
 * @brief Recursively normalizes a SerialParallelDecomposition.
 *
 * @details This function performs the following semantic substitutions:
 * - Deletes every empty SerialSplit and ParallelSplit item:
 *   S(P(S()), Node(1), Node(2)) -> S(Node(1), Node(2))
 *
 * - Replaces SerialSplit and ParallelSplit of size 1 with their content:
 *   S(S(Node(1)), P(Node(2))) -> S(Node(1), Node(2))
 *
 */
SerialParallelDecomposition normalize(SerialParallelDecomposition const &sp);

} // namespace FlexFlow

#endif
