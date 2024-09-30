#include "compiler/machine_mapping/machine_mapping_result_tree/machine_mapping_result_tree.h"
#include "compiler/machine_mapping/machine_mapping_result_tree/mm_result_tree_series_split.h"
#include "compiler/machine_mapping/machine_mapping_result_tree/mm_result_tree_parallel_split.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/make.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/generic_binary_sp_decomposition_tree/get_node_type.h"

namespace FlexFlow {

SPDecompositionTreeNodeType get_node_type(MachineMappingResultTree const &t) {
  return get_node_type(t.raw_tree);
}

float get_mm_result_tree_cost(MachineMappingResultTree const &t) {
  return visit<float>(
    t,
    overload {
      [](MMResultTreeSeriesSplit const &series) {
        return get_cost(series);
      },
      [](MMResultTreeParallelSplit const &parallel) {
        return get_cost(parallel);
      },
      [](MMResultTreeLeafLabel const &leaf) {
        return leaf.cost;
      },
    });
}

MachineMappingResultTree make_series_split(float comm_cost,
                                           BinaryTreePathEntry problem_tree_path_entry,
                                           MachineMappingResultTree const &pre,
                                           MachineMappingResultTree const &post) {
  MMResultTreeSeriesSplitLabel label = MMResultTreeSeriesSplitLabel{
    /*cost=*/get_mm_result_tree_cost(pre) + comm_cost + get_mm_result_tree_cost(post),
    /*problem_tree_path_entry=*/problem_tree_path_entry,
  };

  return MachineMappingResultTree{
    make_generic_binary_series_split(label, pre.raw_tree, post.raw_tree),
  };
}

MachineMappingResultTree make_parallel_split(MachineMappingResultTree const &lhs,
                                             MachineMappingResultTree const &rhs) {
  MMResultTreeParallelSplitLabel label = MMResultTreeParallelSplitLabel{
    /*cost=*/std::max(get_mm_result_tree_cost(lhs), get_mm_result_tree_cost(rhs)),
    /*problem_tree_path_entry=*/problem_tree_path_entry,
  };

  return MachineMappingResultTree{
    make_generic_binary_series_split(label, pre.raw_tree, post.raw_tree),
  };
}

MachineMappingResultTree make_leaf_node(float cost, MachineView const &) {

}

} // namespace FlexFlow
