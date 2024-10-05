#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GET_NUM_TREE_NODES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GET_NUM_TREE_NODES_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_left_child.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_right_child.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/visit.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
int get_num_tree_nodes(GenericBinarySPDecompositionTree<SeriesLabel,
                                                        ParallelLabel,
                                                        LeafLabel> const &tt) {
  return visit<int>(tt,
                    overload{
                        [](LeafLabel const &t) { return 1; },
                        [](GenericBinarySeriesSplit<SeriesLabel,
                                                    ParallelLabel,
                                                    LeafLabel> const &s) {
                          return get_num_tree_nodes(s);
                        },
                        [](GenericBinaryParallelSplit<SeriesLabel,
                                                      ParallelLabel,
                                                      LeafLabel> const &p) {
                          return get_num_tree_nodes(p);
                        },
                    });
}

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
int get_num_tree_nodes(
    GenericBinarySeriesSplit<SeriesLabel, ParallelLabel, LeafLabel> const &s) {
  return 1 + get_num_tree_nodes(get_left_child(s)) +
         get_num_tree_nodes(get_right_child(s));
}

template <typename SeriesLabel, typename ParallelLabel, typename LeafLabel>
int get_num_tree_nodes(
    GenericBinaryParallelSplit<SeriesLabel, ParallelLabel, LeafLabel> const
        &p) {
  return 1 + get_num_tree_nodes(get_left_child(p)) +
         get_num_tree_nodes(get_right_child(p));
}

} // namespace FlexFlow

#endif
