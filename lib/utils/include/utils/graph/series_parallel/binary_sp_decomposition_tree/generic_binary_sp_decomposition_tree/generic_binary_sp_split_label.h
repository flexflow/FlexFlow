#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_SPLIT_LABEL_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_SERIES_PARALLEL_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_DECOMPOSITION_TREE_GENERIC_BINARY_SP_SPLIT_LABEL_H

#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/generic_binary_sp_split_label.dtg.h"
#include "utils/graph/series_parallel/sp_decomposition_tree_node_type.dtg.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename SeriesLabel, typename ParallelLabel>
SPDecompositionTreeNodeType get_node_type(GenericBinarySPSplitLabel<SeriesLabel, ParallelLabel> const &label) {
  return label.template visit<SPDecompositionTreeNodeType>(overload {
    [](GenericBinarySeriesSplitLabel<SeriesLabel> const &) { return SPDecompositionTreeNodeType::SERIES; },
    [](GenericBinaryParallelSplitLabel<ParallelLabel> const &) { return SPDecompositionTreeNodeType::PARALLEL; },
  });
}

template <typename SeriesLabel, typename ParallelLabel>
GenericBinarySPSplitLabel<SeriesLabel, ParallelLabel> make_generic_binary_series_split_label(SeriesLabel const &label) {
  return GenericBinarySPSplitLabel<SeriesLabel, ParallelLabel>{
    GenericBinarySeriesSplitLabel<SeriesLabel>{
      label, 
    },
  };
}

template <typename SeriesLabel, typename ParallelLabel>
GenericBinarySPSplitLabel<SeriesLabel, ParallelLabel> make_generic_binary_parallel_split_label(ParallelLabel const &label) {
  return GenericBinarySPSplitLabel<SeriesLabel, ParallelLabel>{
    GenericBinaryParallelSplitLabel<ParallelLabel>{
      label, 
    },
  };
}

template <typename SeriesLabel, typename ParallelLabel>
SeriesLabel require_generic_binary_series_split_label(GenericBinarySPSplitLabel<SeriesLabel, ParallelLabel> const &label) {
  return label.template get<GenericBinarySeriesSplitLabel<SeriesLabel>>().raw_label;
}

template <typename SeriesLabel, typename ParallelLabel>
ParallelLabel require_generic_binary_parallel_split_label(GenericBinarySPSplitLabel<SeriesLabel, ParallelLabel> const &label) {
  return label.template get<GenericBinaryParallelSplitLabel<ParallelLabel>>().raw_label;
}

} // namespace FlexFlow

#endif
