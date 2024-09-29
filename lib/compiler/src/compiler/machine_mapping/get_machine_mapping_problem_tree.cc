#include "compiler/machine_mapping/get_machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/get_abstracted_tensor_set_movement_across_split.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "compiler/series_parallel/pcg_binary_parallel_split.h"
#include "compiler/series_parallel/pcg_binary_series_split.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.dtg.h"
#include "compiler/series_parallel/pcg_binary_sp_decomposition.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/overload.h"

namespace FlexFlow {

MachineMappingProblemTree get_machine_mapping_problem_tree(ParallelComputationGraph const &pcg,
                                                           PCGBinarySPDecomposition const &sp_decomposition_tree) {
  TransitiveReducedPCG tr_pcg = pcg_get_transitive_reduction(pcg);

  std::function<MachineMappingProblemTree(PCGBinarySPDecomposition const &)> to_problem_tree;  

  to_problem_tree = [&](PCGBinarySPDecomposition const &sp) -> MachineMappingProblemTree {
    return visit<MachineMappingProblemTree>(
      sp,
      overload {
        [&](PCGBinarySeriesSplit const &series) {
          AbstractedTensorSetMovement tensor_movement = get_abstracted_tensor_set_movement_across_split(tr_pcg, series);
          return mm_problem_tree_make_series_split(
            /*tensor_set_movement=*/tensor_movement,
            /*lhs=*/to_problem_tree(get_left_child(series)),
            /*rhs=*/to_problem_tree(get_right_child(series)));
        },
        [&](PCGBinaryParallelSplit const &parallel) {
          return mm_problem_tree_make_parallel_split(
            to_problem_tree(get_left_child(parallel)),
            to_problem_tree(get_right_child(parallel)));
        },
        [&](parallel_layer_guid_t const &leaf) {
          return mm_problem_tree_make_leaf(pcg_get_op_attrs(pcg, leaf));
        }
      });
  };

  return to_problem_tree(sp_decomposition_tree);
}

} // namespace FlexFlow
