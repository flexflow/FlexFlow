#ifndef _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_CRITICAL_PATH_PRESERVING_SP_IZATION_H
#define _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_CRITICAL_PATH_PRESERVING_SP_IZATION_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include <unordered_map>

namespace FlexFlow {

/**
 * @brief Transforms a directed acyclic graph (DAG) into a Serial Parallel (SP)
 * graph. The critical path cost is unchanged, and the SP-ization is done solely
 * through node (work) duplication.
 *
 * @details
 * The resulting graph, encoded as a SerialParallelDecomposition, is a tree
 * whose critical path is the same as that of the original graph. The tree is
 * constructed as follows:
 * - Denote SP(n) as the SerialParallelDecomposition of the subgraph of g whose
 * nodes are all the ancestors of n.
 * - Denote the predecessors of n as M.
 * - Then:
 *   - SP(n) = S(n, P({SP(m) for m in M}))
 *   - SP(root) = root
 *   - SP(sink) = SP(g)
 * Where P, S represent parallel, serial composition respectively.
 *
 * Example:
 * <tt>
 * digraph G {
 *     n1 -> n2;
 *     n1 -> n3;
 *     n2 -> n4;
 *     n2 -> n5;
 *     n3 -> n5;
 *     n5 -> n6;
 *     n4 -> n6;
 * }
 *
 * becomes
 *
 * digraph SP {
 *     n1 [label="n1"];
 *     n2 [label="n2"];
 *     n3 [label="n4"];
 *     n4 [label="n6"];
 *     n5 [label="n1"];
 *     n6 [label="n2"];
 *     n7 [label="n5"];
 *     n8 [label="n1"];
 *     n9 [label="n3"];
 *     n1 -> n2;
 *     n2 -> n3;
 *     n3 -> n4;
 *     n5 -> n6;
 *     n6 -> n7;
 *     n7 -> n4;
 *     n8 -> n9;
 *     n9 -> n7;
 * }
 * </tt>
 *
 * @note g must be a 2 terminal (i.e. single source and single sink) directed
 * acyclic graph.
 */
SerialParallelDecomposition
    critical_path_preserving_sp_ization(DiGraphView const &g);

/**
 * @brief Transforms a directed acyclic graph (DAG) into a Serial Parallel (SP)
 * graph with coalescing. The critical path cost is unchanged, and the
 * SP-ization is done solely through node (work) duplication.
 *
 * @details
 * This SP-ization technique, compared to the previous step, adds an additional
 * coalescing step during parallel composition to reduce node duplication. The
 * recursive formulation is equivalent, but the parallelization performs an
 * additional coalescing step, where parallel strands with common heads are
 * merged together. Example: P(S(1,2), S(1,3)) -> P(1, S(2,3)).
 *
 * Example:
 * <tt>
 * digraph G {
 *     n1 -> n2;
 *     n1 -> n3;
 *     n2 -> n4;
 *     n2 -> n5;
 *     n3 -> n5;
 *     n5 -> n6;
 *     n4 -> n6;
 * }
 *
 * becomes
 *
 * digraph SP {
 *     n1 [label="n1"];
 *     n2 [label="n2"];
 *     n3 [label="n4"];
 *     n4 [label="n6"];
 *     n6 [label="n2"];
 *     n7 [label="n5"];
 *     n9 [label="n3"];
 *     n1 -> n2;
 *     n2 -> n3;
 *     n3 -> n4;
 *     n1 -> n6;
 *     n6 -> n7;
 *     n7 -> n4;
 *     n1 -> n9;
 *     n9 -> n7;
 * }
 * </tt>
 */
SerialParallelDecomposition
    critical_path_preserving_sp_ization_with_coalescing(DiGraphView const &g);

} // namespace FlexFlow

#endif
