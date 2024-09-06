#ifndef _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_WORK_PRESERVING_SP_IZATION_H
#define _FLEXFLOW_UTILS_GRAPH_SERIAL_PARALLEL_WORK_PRESERVING_SP_IZATION_H

#include "utils/graph/digraph/digraph_view.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include <unordered_map>

namespace FlexFlow {

/**
 * @brief
 * Transforms a directed acyclic graph (DAG) into a Serial Parallel (SP)
 * graph. The total number of nodes remains unchanged, and the SP-ization is
 *done solely through edge (dependency) duplication.
 * @details
 * The graph is first partitioned into strata: the i_th stratum contains all the
 *nodes whose critical path length has length i. The nodes in a given stratum
 *are composed in parallel, and the strata are serially composed in succession.
 *
 * Example:
 *
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
 * becomes
 *
 * digraph SP {
 *     n1 -> n2;
 *     n1 -> n3;
 *     n2 -> n4;
 *     n2 -> n5;
 *     n3 -> n5;
 *     n4 -> n6;
 *     n5 -> n6;
 *     n4 -> n6;
 * }
 * </tt>
 * @note g must be a directed acyclic graph.
 **/
SerialParallelDecomposition stratum_sync_sp_ization(DiGraphView const &g);

/**
 * @brief
 * Transforms a directed acyclic graph (DAG) into a Serial Parallel (SP)
 * graph. The total number of nodes remains unchanged, and the SP-ization is
 *done solely through edge (dependency) duplication.
 *
 * @details
 * The algorithm operates under the same principles as
 *`stratum_sync_sp_ization`: that is, a stratification step where the nodes are
 *partitioned into strata, followed by a merging step where the strata are
 *joined. The difference concerns the stratification step, which is cost-aware,
 *so that the different disjoint subgraphs present within the same strata have a
 *similar critical path cost, thus minimizing the overall critical path cost of
 *the SP-ized graph.
 **/
SerialParallelDecomposition cost_aware_stratum_sync_sp_ization(
    DiGraphView const &g, std::unordered_map<Node, float> const &cost_map);

} // namespace FlexFlow

#endif
