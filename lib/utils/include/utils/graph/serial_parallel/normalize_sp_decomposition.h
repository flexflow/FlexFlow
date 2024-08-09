#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_normalize_sp_decomposition_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_normalize_sp_decomposition_H

#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"

namespace FlexFlow {

/**
 * @brief Recursively normalizes a SerialParallelDecomposition.
 *
 * @details This function performs the following semantic substitutions:
 * - Deletes every empty SerialSplit and ParallelSplit item, e.g.,
 *   <tt>S(P(S()), Node(1), Node(2)) -> S(Node(1), Node(2))</tt>
 *
 * - Replaces SerialSplit and ParallelSplit of size 1 with their content, e.g.,
 *   <tt>S(S(Node(1)), P(Node(2))) -> S(Node(1), Node(2))</tt>)
 *
 */
SerialParallelDecomposition
    normalize_sp_decomposition(SerialParallelDecomposition const &sp);

} // namespace FlexFlow

#endif
