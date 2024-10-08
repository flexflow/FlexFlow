#include "compiler/machine_mapping/machine_mapping_problem_tree/get_machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/get_abstracted_tensor_set_movement_across_split.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/overload.h"

namespace FlexFlow {

MachineMappingProblemTree get_machine_mapping_problem_tree(
    ParallelComputationGraph const &pcg,
    PCGBinarySPDecomposition const &sp_decomposition_tree) {
  TransitiveReducedPCG tr_pcg = pcg_get_transitive_reduction(pcg);

  std::function<MachineMappingProblemTree(PCGBinarySPDecomposition const &)>
      to_problem_tree;

  to_problem_tree =
      [&](PCGBinarySPDecomposition const &sp) -> MachineMappingProblemTree {
    return sp.visit<MachineMappingProblemTree>(overload{
        [&](PCGBinarySeriesSplit const &series) {
          AbstractedTensorSetMovement tensor_movement =
              get_abstracted_tensor_set_movement_across_split(tr_pcg, series);
          return MachineMappingProblemTree{
              MMProblemTreeSeriesSplit{
                  /*tensor_set_movement=*/tensor_movement,
                  /*lhs=*/to_problem_tree(series.get_left_child()),
                  /*rhs=*/to_problem_tree(series.get_right_child()),
              },
          };
        },
        [&](PCGBinaryParallelSplit const &parallel) {
          return MachineMappingProblemTree{
              MMProblemTreeParallelSplit{
                  to_problem_tree(parallel.get_left_child()),
                  to_problem_tree(parallel.get_right_child()),
              },
          };
        },
        [&](parallel_layer_guid_t const &leaf) {
          return MachineMappingProblemTree{
              get_unmapped_op_cost_estimate_key_for_layer(pcg, leaf),
          };
        },
    });
  };

  return to_problem_tree(sp_decomposition_tree);
}

} // namespace FlexFlow
